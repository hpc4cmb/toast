
// Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/sys_utils.hpp>
#include <toast/math_fft.hpp>
#include <toast/math_rng.hpp>
#include <toast/tod_simnoise.hpp>

#include <sstream>


void tod_sim_noise_psd_interp(double rate, int64_t samples, int64_t oversample,
                              int64_t n_batch, int64_t n_binned,
                              double const * binned_freq, double const * binned_psds,
                              int64_t & fftlen,
                              toast::AlignedVector <double> & interp_psds) {
    fftlen = 2;
    while (fftlen <= (oversample * samples)) {
        fftlen *= 2;
    }
    int64_t psdlen = (fftlen / 2) + 1;
    double norm = rate * static_cast <double> (psdlen - 1);
    double increment = rate / static_cast <double> (fftlen - 1);

    if (binned_freq[0] > increment) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o << "input PSDs have lowest frequency " << binned_freq[0]
          << "Hz, which does not allow interpolation to " << increment
          << "Hz";
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    double nyquist = 0.5 * rate;
    if (::fabs((binned_freq[n_binned - 1] - nyquist) / nyquist) > 0.01) {
        auto here = TOAST_HERE();
        auto log = toast::Logger::get();
        std::ostringstream o;
        o.precision(16);
        o << "last frequency element does not match Nyquist "
          << "frequency for given sample rate: "
          << binned_freq[n_binned - 1] << " != " << nyquist;
        log.error(o.str().c_str(), here);
        throw std::runtime_error(o.str().c_str());
    }

    // Compute the (common) log frequency products

    toast::AlignedVector <double> logfreq(n_binned);

    double freqshift = increment;
    if (toast::is_aligned(binned_freq)) {
        #pragma omp simd
        for (int64_t i = 0; i < n_binned; ++i) {
            logfreq[i] = ::log10(binned_freq[i] + freqshift);
        }
    } else {
        for (int64_t i = 0; i < n_binned; ++i) {
            logfreq[i] = ::log10(binned_freq[i] + freqshift);
        }
    }

    toast::AlignedVector <double> stepinv(n_binned);
    for (int64_t ibin = 0; ibin < n_binned - 1; ++ibin) {
        stepinv[ibin] = 1 / (logfreq[ibin + 1] - logfreq[ibin]);
    }

    // Resize the output
    interp_psds.resize(n_batch * psdlen);

    // Process all psds
    #pragma omp parallel default(shared)
    {
        toast::AlignedVector <double> logpsd(n_binned);

        #pragma omp for schedule(static)
        for (int64_t ibatch = 0; ibatch < n_batch; ++ibatch) {
            int64_t offset_binned = ibatch * n_binned;
            int64_t offset_psd = ibatch * psdlen;

            double psdmin = 1e30;
            for (int64_t i = 0; i < n_binned; ++i) {
                if (binned_psds[offset_binned + i] != 0) {
                    if (binned_psds[offset_binned + i] < psdmin) {
                        psdmin = binned_psds[offset_binned + i];
                    }
                }
            }

            if (psdmin < 0) {
                auto here = TOAST_HERE();
                auto log = toast::Logger::get();
                std::string msg("input PSD values should be >= zero");
                log.error(msg.c_str(), here);
                throw std::runtime_error(msg.c_str());
            }

            // Perform a logarithmic interpolation.  In order to avoid zero
            // values, we shift the PSD by a fixed amount in frequency and
            // amplitude.

            double psdshift = 0.01 * psdmin;

            if (toast::is_aligned(binned_psds)) {
                #pragma omp simd
                for (int64_t i = 0; i < n_binned; ++i) {
                    logpsd[i] = ::log10(::sqrt(
                                            binned_psds[offset_binned + i] * norm) +
                                        psdshift);
                }
            } else {
                for (int64_t i = 0; i < n_binned; ++i) {
                    logpsd[i] = ::log10(::sqrt(
                                            binned_psds[offset_binned + i] * norm) +
                                        psdshift);
                }
            }

            #pragma omp simd
            for (int64_t i = 0; i < psdlen; ++i) {
                interp_psds[offset_psd +
                            i] = ::log10(
                    increment * static_cast <double> (i) + freqshift);
            }

            int64_t ibin = 0;

            #pragma omp simd
            for (int64_t i = 0; i < psdlen; ++i) {
                double loginterp_freq = interp_psds[offset_psd + i];
                while ((ibin < (n_binned - 2)) &&
                       (logfreq[ibin + 1] < loginterp_freq)) {
                    ++ibin;
                }
                double r = (loginterp_freq - logfreq[ibin]) * stepinv[ibin];
                interp_psds[offset_psd + i] = logpsd[ibin] + r *
                                              (logpsd[ibin + 1] - logpsd[ibin]);
                interp_psds[offset_psd + i] = pow(10, interp_psds[offset_psd + i]);
                interp_psds[offset_psd + i] -= psdshift;
            }

            // Zero out DC value
            interp_psds[offset_psd + 0] = 0;
        }
    }

    return;
}

void toast::tod_sim_noise_timestream(
    uint64_t realization, uint64_t telescope, uint64_t component,
    uint64_t obsindx, uint64_t detindx, double rate, int64_t firstsamp,
    int64_t samples, int64_t oversample, const double * freq,
    const double * psd, int64_t psdlen, double * noise) {
    // Generate a single noise timestream.
    //
    // Use the RNG parameters to generate unit-variance Gaussian samples and then modify
    //     the Fourier domain amplitudes to match the desired PSD.
    //
    // The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
    // which each consist of two unsigned 64bit integers.  These four numbers together
    //     uniquely identify a single sample.  We construct those four numbers in the
    //     following way:
    //
    // key1 = realization * 2^32 + telescope * 2^16 + component key2 = obsindx * 2^32 +
    //     detindx counter1 = currently unused (0) counter2 = sample in stream
    //
    // counter2 is incremented internally by the RNG function as it calls the underlying
    //     Random123 library for each sample.

    int64_t fftlen;
    toast::AlignedVector <double> interp_psd;

    tod_sim_noise_psd_interp(rate, samples, oversample, 1, psdlen, freq, psd, fftlen,
                             interp_psd);

    int64_t npsd = (fftlen / 2) + 1;

    // Get a plan of the correct size and direction from the global
    // per-process plan store.

    auto & store = toast::FFTPlanReal1DStore::get();
    auto plan = store.backward(fftlen, 1);

    double * pdata = plan->fdata(0);

    // gaussian Re/Im randoms, packed into a half-complex array

    uint64_t key1 = realization * 4294967296 + telescope * 65536 + component;
    uint64_t key2 = obsindx * 4294967296 + detindx;
    uint64_t counter1 = 0;
    uint64_t counter2 = static_cast <uint64_t> (firstsamp * oversample);

    toast::rng_dist_normal(fftlen, key1, key2, counter1, counter2, pdata);

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

void toast::tod_sim_noise_timestream_batch(
    uint64_t realization, uint64_t telescope, uint64_t component,
    uint64_t obsindx, double rate, int64_t firstsamp,
    int64_t samples, int64_t oversample, int64_t ndet, uint64_t * detindices,
    int64_t psdlen, const double * freq, const double * psds, double * noise) {
    // Generate multiple noise timestreams at once.
    //
    // Use the RNG parameters to generate unit-variance Gaussian samples and then modify
    //     the Fourier domain amplitudes to match the desired PSD.
    //
    // The RNG (Threefry2x64 from Random123) takes a "key" and a "counter"
    // which each consist of two unsigned 64bit integers.  These four numbers together
    //     uniquely identify a single sample.  We construct those four numbers in the
    //     following way:
    //
    // key1 = realization * 2^32 + telescope * 2^16 + component key2 = obsindx * 2^32 +
    //     detindx counter1 = currently unused (0) counter2 = sample in stream
    //
    // counter2 is incremented internally by the RNG function as it calls the underlying
    //     Random123 library for each sample.
    //
    // The frequency vector is common to all detectors.  The detindices are the RNG
    // index for each detector.  The psd vector contains the flat-packed psds for
    // all detectors.  The output noise vector contains the flat-packed timestreams
    // for all detectors.

    int64_t fftlen;
    toast::AlignedVector <double> interp_psds;

    tod_sim_noise_psd_interp(rate, samples, oversample, ndet, psdlen, freq, psds,
                             fftlen, interp_psds);

    int64_t npsd = (fftlen / 2) + 1;

    // Get the plan for this batch size and length.

    auto & store = toast::FFTPlanReal1DStore::get();
    auto plan = store.backward(fftlen, ndet);

    // Populate the Fourier domain buffers

    #pragma omp parallel for default(shared) schedule(static)
    for (int64_t idet = 0; idet < ndet; ++idet) {
        int64_t offset_psd = idet * npsd;

        double * pdata = plan->fdata(idet);

        // Gaussian Re/Im randoms, packed into a half-complex array
        uint64_t key1 = realization * 4294967296 + telescope * 65536 + component;
        uint64_t key2 = obsindx * 4294967296 + detindices[idet];
        uint64_t counter1 = 0;
        uint64_t counter2 = static_cast <uint64_t> (firstsamp * oversample);

        toast::rng_dist_normal(fftlen, key1, key2, counter1, counter2, pdata);

        pdata[0] *= interp_psds[offset_psd + 0];
        for (int64_t i = 1; i < (fftlen / 2); ++i) {
            double psdval = interp_psds[offset_psd + i];
            pdata[i] *= psdval;
            pdata[fftlen - i] *= psdval;
        }

        pdata[fftlen / 2] *= interp_psds[offset_psd + npsd - 1];
    }

    // Inverse FFT
    plan->exec();

    // Copy data to output
    for (int64_t idet = 0; idet < ndet; ++idet) {
        int64_t offset = (fftlen - samples) / 2;
        int64_t offset_nse = idet * samples;
        double * pdata = plan->tdata(idet) + offset;
        std::copy(pdata, (pdata + samples), &(noise[offset_nse]));

        // subtract the DC level

        double DC = 0;
        for (int64_t i = 0; i < samples; ++i) {
            DC += noise[offset_nse + i];
        }
        DC /= static_cast <double> (samples);

        for (int64_t i = 0; i < samples; ++i) {
            noise[offset_nse + i] -= DC;
        }
    }

    return;
}
