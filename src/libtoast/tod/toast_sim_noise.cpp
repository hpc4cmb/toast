/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_tod_internal.hpp>


void toast::sim_noise::sim_noise_timestream(
    const uint64_t realization, const uint64_t telescope,
    const uint64_t component, const uint64_t obsindx, const uint64_t detindx,
    const double rate, const uint64_t firstsamp, const uint64_t samples,
    const uint64_t oversample, const double *freq, const double *psd,
    const uint64_t psdlen, double *noise) {

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

    uint64_t fftlen = 2;
    while (fftlen <= (oversample * samples)) {
        fftlen *= 2;
    }
    uint64_t npsd = fftlen/2 + 1;
    double norm = rate * float(npsd - 1);

    // Python implementation is missing "-1"
    double increment = rate / (fftlen - 1);

    for (uint64_t i=0; i<psdlen; ++i) {
        if (psd[i] < 0) {
            throw std::runtime_error("input PSD values should be >= zero");
        }
    }

    if (freq[0] > increment) {
        throw std::runtime_error(
            "input PSD does not go to low enough frequency to "
            "allow for interpolation");
    }

    double nyquist = rate / 2;
    if (abs((freq[psdlen-1]-nyquist) / nyquist) > .01) {
        std::ostringstream o;
        o.precision(16);
        o << "last frequency element does not match Nyquist "
          << "frequency for given sample rate: "
          << freq[psdlen-1] << " != " << nyquist
          << std::endl;
        throw std::runtime_error(o.str().c_str());
    }

    // Perform a logarithmic interpolation.  In order to avoid zero
    // values, we shift the PSD by a fixed amount in frequency and
    // amplitude.

    double psdmin = 1e30;
    for (uint64_t i=0; i<psdlen; ++i) {
        if (psd[i] != 0) {
            if (psd[i] < psdmin) psdmin = psd[i];
        }
    }

    double psdshift = 0.01 * psdmin;
    double freqshift = increment;

    std::vector<double> logfreq(psdlen);
    std::vector<double> logpsd(psdlen);
    for (uint64_t i=0; i<psdlen; ++i) {
        logfreq[i] = log10(freq[i] + freqshift);
        logpsd[i] = log10(psd[i] + psdshift);
    }

    uint64_t ibin = 0;
    std::vector<double> interp_psd(npsd);
    for (uint64_t i=0; i<npsd; ++i) {
        double loginterp_freq = log10(i*increment + freqshift);
        while (ibin < psdlen-1 && logfreq[ibin+1] < loginterp_freq) ++ibin;
        double r = (loginterp_freq-logfreq[ibin])
            / (logfreq[ibin+1]-logfreq[ibin]);
        double loginterp_psd = (1-r)*logpsd[ibin] + r*logfreq[ibin+1];
        interp_psd[i] = pow(10, loginterp_psd) - psdshift;
        //std::cerr<<"interp_psd[" << i << "] = " << interp_psd[i] << std::endl; // DEBUG
    }

    //scale = np.sqrt(interp_psd * norm)

    // Zero out DC value

    interp_psd[0] = 0;

    // gaussian Re/Im randoms, packed into a complex valued array

    uint64_t key1 = realization*4294967296 + telescope*65536 + component;
    uint64_t key2 = obsindx*4294967296 + detindx;
    uint64_t counter1 = 0;
    uint64_t counter2 = firstsamp * oversample;
    std::vector<double> rngdata(2*npsd);

    rng::dist_normal(2*npsd, key1, key2, counter1, counter2, rngdata.data());

    fft::r1d_p plan(fft::r1d::create(fftlen, 1, fft::plan_type::fast,
                                     fft::direction::backward, 1));

    for (uint64_t i=0; i<npsd; ++i) plan->fdata()[0][i] = rngdata[i];
    for (uint64_t i=0; i<npsd; ++i) plan->fdata()[0][fftlen-1-i] = rngdata[npsd+i];

    plan->fdata()[0][0] = 0;
    for (uint64_t i=1; i<=fftlen/2; ++i) {
        double psdval = interp_psd[i] * norm;
        plan->fdata()[0][i] *= psdval;
        plan->fdata()[0][fftlen-i] *= psdval;
    }

    plan->exec();

    uint64_t offset = (fftlen - samples) / 2;

    for (long i=0; i<samples; ++i) {
        noise[i] = plan->tdata()[0][offset+i];
    }

    // subtract the DC level

    double DC = 0;
    for (long i=0; i<samples; ++i) DC += noise[i];
    DC /= samples;
    for (long i=0; i<samples; ++i) noise[i] -= DC;

    return;

}
