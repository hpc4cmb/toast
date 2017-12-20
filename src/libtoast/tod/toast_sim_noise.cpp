/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_tod_internal.hpp>
#include <toast_util_internal.hpp>

void toast::sim_noise::sim_noise_timestream(
    const uint64_t realization, const uint64_t telescope,
    const uint64_t component, const uint64_t obsindx, const uint64_t detindx,
    const double rate, const uint64_t firstsamp, const uint64_t samples,
    const uint64_t oversample, const double *freq, const double *psd,
    const uint64_t psdlen, double *noise) {

    TOAST_AUTO_TIMER();

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

    //double t0 = MPI_Wtime(); // DEBUG

    uint64_t fftlen = 2;
    while (fftlen <= (oversample * samples)) fftlen *= 2;
    uint64_t npsd = fftlen/2 + 1;
    double norm = rate * float(npsd - 1);

    // Python implementation is missing "-1"
    double increment = rate / (fftlen - 1);

    if (freq[0] > increment) {
        throw std::runtime_error(
            "input PSD does not go to low enough frequency to "
            "allow for interpolation");
    }

    double psdmin = 1e30;
    for (uint64_t i=0; i<psdlen; ++i) {
        if (psd[i] != 0) {
            if (psd[i] < psdmin) psdmin = psd[i];
        }
    }

    if (psdmin < 0) {
        throw std::runtime_error("input PSD values should be >= zero");
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

    double psdshift = 0.01 * psdmin;
    double freqshift = increment;

    std::vector<double> logfreq(psdlen);
    std::vector<double> logpsd(psdlen);
    for (uint64_t i=0; i<psdlen; ++i) {
        logfreq[i] = log10(freq[i] + freqshift);
        logpsd[i] = log10(sqrt(psd[i]*norm) + psdshift);
    }

    std::vector<double> interp_psd(npsd);
    for (uint64_t i=0; i<npsd; ++i) {
        interp_psd[i] = log10(i*increment + freqshift);
    }

    std::vector<double> stepinv(psdlen);
    for (uint64_t ibin=0; ibin<psdlen-1; ++ibin) {
        stepinv[ibin] = 1 / (logfreq[ibin+1]-logfreq[ibin]);
    }

    //double t1 = MPI_Wtime(); // DEBUG
    //std::cerr << std::endl
    //          << "noise sim init took     " << t1 - t0 << " s" << std::endl; // DEBUG

    uint64_t ibin = 0;
    for (uint64_t i=0; i<npsd; ++i) {
        double loginterp_freq = interp_psd[i];

        while (ibin < psdlen-2 && logfreq[ibin+1] < loginterp_freq) ++ibin;

        double r = (loginterp_freq-logfreq[ibin]) * stepinv[ibin];
        interp_psd[i] = logpsd[ibin] + r*(logpsd[ibin+1]-logpsd[ibin]);
    }

    logfreq.clear();
    logpsd.clear();
    stepinv.clear();

    for (uint64_t i=0; i<npsd; ++i) interp_psd[i] = pow(10, interp_psd[i]);

    for (uint64_t i=0; i<npsd; ++i) interp_psd[i] -= psdshift;

    //double t2 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim interp took   " << t2 - t1 << " s" << std::endl; // DEBUG

    // Zero out DC value

    interp_psd[0] = 0;

    // gaussian Re/Im randoms, packed into a half-complex array

    uint64_t key1 = realization*4294967296 + telescope*65536 + component;
    uint64_t key2 = obsindx*4294967296 + detindx;
    uint64_t counter1 = 0;
    uint64_t counter2 = firstsamp * oversample;
    std::vector<double> rngdata(fftlen);

    rng::dist_normal(fftlen, key1, key2, counter1, counter2, rngdata.data());

    //double t3 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim rng took      " << t3 - t2 << " s" << std::endl; // DEBUG

    //fft::r1d_p plan(fft::r1d::create(fftlen, 1, fft::plan_type::fast,
    //                                 fft::direction::backward, 1));
    fft::r1d_plan_store store = fft::r1d_plan_store::get();
    fft::r1d_p plan = store.backward(fftlen, 1);

    //double t3p1 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim fft plan took " << t3p1 - t3 << " s" << std::endl; // DEBUG

    double *pdata = plan->fdata()[0];

    memcpy(pdata, rngdata.data(), sizeof(double)*fftlen);

    pdata[0] *= interp_psd[0];
    for (uint64_t i=1; i<fftlen/2; ++i) {
        double psdval = interp_psd[i];
        pdata[i] *= psdval;
        pdata[fftlen-i] *= psdval;
    }
    pdata[fftlen/2] *= interp_psd[npsd-1];

    //double t4 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim FFT prep took " << t4 - t3p1 << " s" << std::endl; // DEBUG

    plan->exec();

    //double t5 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim FFT took      " << t5 - t4 << " s" << std::endl; // DEBUG

    uint64_t offset = (fftlen - samples) / 2;
    pdata = plan->tdata()[0] + offset;
    memcpy(noise, pdata, sizeof(double)*samples);

    // subtract the DC level

    double DC = 0;
    for (uint64_t i=0; i<samples; ++i) DC += noise[i];
    DC /= samples;
    for (uint64_t i=0; i<samples; ++i) noise[i] -= DC;

    //double t6 = MPI_Wtime(); // DEBUG
    //std::cerr << "noise sim finalize took " << t6 - t5 << " s" << std::endl; // DEBUG
    //exit(-1); // DEBUG

    return;

}
