
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_rng.hpp>
#include <toast/tod_simnoise.hpp>

#include <sstream>


void toast::tod_sim_noise_timestream(
    uint64_t realization, uint64_t telescope, uint64_t component,
    uint64_t obsindx, uint64_t detindx, double rate, int64_t firstsamp,
    int64_t samples, int64_t oversample, const double * freq,
    const double * psd, int64_t psdlen, double * noise) {
    /*
       Generate a noise timestream, given a starting RNG state.

       Use the RNG parameters to generate unit-variance Gaussian samples
       and then modify the Fourier domain amplitudes to match the desired
       PSD.

       The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
       which each consist of two unsigned 64bit integers.  These four
       numbers together uniquely identify a single sample.  We construct
       those four numbers in the following way:

       key1 = realization * 2^32 + telescope * 2^16 + component
       key2 = obsindx * 2^32 + detindx
       counter1 = currently unused (0)
       counter2 = sample in stream

       counter2 is incremented internally by the RNG function as it calls
       the underlying Random123 library for each sample.

       Args:
        realization (int): the Monte Carlo realization.
        telescope (int): a unique index assigned to a telescope.
        component (int): a number representing the type of timestream
            we are generating (detector noise, common mode noise,
            atmosphere, etc).
        obsindx (int): the global index of this observation.
        detindx (int): the global index of this detector.
        rate (float): the sample rate.
        firstsamp (int): the start sample in the stream.
        samples (int): the number of samples to generate.
        oversample (int): the factor by which to expand the FFT length
            beyond the number of samples.
        freq (array): the frequency points of the PSD.
        psd (array): the PSD values.

       Returns (tuple):
        the timestream array, the interpolated PSD frequencies, and
            the interpolated PSD values.
     */

    int64_t fftlen = 2;
    while (fftlen <= (oversample * samples)) fftlen *= 2;
    int64_t npsd = (fftlen / 2) + 1;
    double norm = rate * static_cast <double> (npsd - 1);

    double increment = rate / static_cast <double> (fftlen - 1);

    if (freq[0] > increment) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "input PSD has lowest frequency " << freq[0]
          << "Hz, which does not allow interpolation to " << increment
          << "Hz";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    double psdmin = 1e30;
    for (int64_t i = 0; i < psdlen; ++i) {
        if (psd[i] != 0) {
            if (psd[i] < psdmin) psdmin = psd[i];
        }
    }

    if (psdmin < 0) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::string msg("input PSD values should be >= zero");
        log.error(msg.c_str(), here);
        throw std::runtime_error(msg.c_str());
    }

    double nyquist = 0.5 * rate;
    if (::fabs((freq[psdlen - 1] - nyquist) / nyquist) > 0.01) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o.precision(16);
        o << "last frequency element does not match Nyquist "
          << "frequency for given sample rate: "
          << freq[psdlen - 1] << " != " << nyquist;
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    // Perform a logarithmic interpolation.  In order to avoid zero
    // values, we shift the PSD by a fixed amount in frequency and
    // amplitude.

    double psdshift = 0.01 * psdmin;
    double freqshift = increment;

    toast::AlignedVector <double> logfreq(psdlen);
    toast::AlignedVector <double> logpsd(psdlen);

    if (toast::is_aligned(freq) && toast::is_aligned(psd)) {
        #pragma omp simd
        for (int64_t i = 0; i < psdlen; ++i) {
            logfreq[i] = ::log10(freq[i] + freqshift);
            logpsd[i] = ::log10(::sqrt(psd[i] * norm) + psdshift);
        }
    } else {
        for (int64_t i = 0; i < psdlen; ++i) {
            logfreq[i] = ::log10(freq[i] + freqshift);
            logpsd[i] = ::log10(::sqrt(psd[i] * norm) + psdshift);
        }
    }

    toast::AlignedVector <double> interp_psd(npsd);

    #pragma omp simd
    for (int64_t i = 0; i < npsd; ++i) {
        interp_psd[i] = ::log10(increment * static_cast <double> (i) +
                                freqshift);
    }

    toast::AlignedVector <double> stepinv(psdlen);
    for (int64_t ibin = 0; ibin < psdlen - 1; ++ibin) {
        stepinv[ibin] = 1 / (logfreq[ibin + 1] - logfreq[ibin]);
    }

    int64_t ibin = 0;

    #pragma omp simd
    for (int64_t i = 0; i < npsd; ++i) {
        double loginterp_freq = interp_psd[i];
        while ((ibin < (psdlen - 2)) && (logfreq[ibin + 1] < loginterp_freq)) {
            ++ibin;
        }
        double r = (loginterp_freq - logfreq[ibin]) * stepinv[ibin];
        interp_psd[i] = logpsd[ibin] + r * (logpsd[ibin + 1] - logpsd[ibin]);
        interp_psd[i] = pow(10, interp_psd[i]);
        interp_psd[i] -= psdshift;
    }

    // Zero out DC value
    interp_psd[0] = 0;

    // gaussian Re/Im randoms, packed into a half-complex array

    uint64_t key1 = realization * 4294967296 + telescope * 65536 + component;
    uint64_t key2 = obsindx * 4294967296 + detindx;
    uint64_t counter1 = 0;
    uint64_t counter2 = static_cast <uint64_t> (firstsamp * oversample);
    toast::AlignedVector <double> rngdata(fftlen);

    toast::rng_dist_normal(fftlen, key1, key2, counter1, counter2,
                           rngdata.data());

    // Get a plan of the correct size and direction from the global
    // per-process plan store.

    auto & store = toast::FFTPlanReal1DStore::get();
    auto plan = store.backward(fftlen, 1);

    double * pdata = plan->fdata(0);
    std::copy(rngdata.begin(), rngdata.end(), pdata);

    pdata[0] *= interp_psd[0];
    for (int64_t i = 1; i < (fftlen / 2); ++i) {
        double psdval = interp_psd[i];
        pdata[i] *= psdval;
        pdata[fftlen - i] *= psdval;
    }

    pdata[fftlen / 2] *= interp_psd[npsd - 1];

    plan->exec();

    int64_t offset = (fftlen - samples) / 2;
    pdata = plan->tdata(0) + offset;
    std::copy(pdata, (pdata + samples), noise);

    // subtract the DC level

    double DC = 0;
    for (int64_t i = 0; i < samples; ++i) {
        DC += noise[i];
    }
    DC /= static_cast <double> (samples);

    for (int64_t i = 0; i < samples; ++i) {
        noise[i] -= DC;
    }

    return;
}
