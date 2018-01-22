/*
  Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
  All rights reserved.  Use of this source code is governed by
  a BSD-style license that can be found in the LICENSE file.
*/

#include <toast_atm_internal.hpp>

#include "ATMRefractiveIndexProfile.h"
#include "ATMPercent.h"
#include "ATMPressure.h"
#include "ATMNumberDensity.h"
#include "ATMMassDensity.h"
#include "ATMTemperature.h"
#include "ATMLength.h"
#include "ATMInverseLength.h"
#include "ATMOpacity.h"
#include "ATMHumidity.h"
#include "ATMFrequency.h"
#include "ATMWaterVaporRadiometer.h"
#include "ATMWVRMeasurement.h"
#include "ATMProfile.h"
#include "ATMSpectralGrid.h"
#include "ATMRefractiveIndex.h"
#include "ATMSkyStatus.h"
#include "ATMAngle.h"


atm::AtmProfile get_atmprofile(double altitude, double temperature,
                               double pressure) {

    unsigned int atmType = 1; // Atmospheric type (to reproduce
                              // behavior above the tropopause)
    atm::Temperature T(temperature, "K"); // Ground level temperature (K)
    atm::Length Alt(altitude, "m"); // Altitude of the site (m)
    atm::Length WVL(2, "km"); // Water vapour scale height (km)
    double TLR=-5.6; // Tropospheric lapse rate (K/km)
    atm::Length topAtm(48.0, "km"); // Upper atm. boundary for calculations
    atm::Pressure Pstep(1.0, "mb"); // Primary pressure step
    double PstepFact=1.2; // Pressure step ratio between two
                          // consecutive layers
    atm::Pressure P(pressure, "Pa"); // Pressure (Pa)
    atm::Humidity H(10, "%"); // Placeholder humidity (overridden by PWV)

    return atm::AtmProfile(Alt, P, T, TLR, H, WVL, Pstep, PstepFact, topAtm, atmType);
}


atm::SkyStatus get_sky_status(double altitude, double temperature,
                              double pressure, double freq) {
    /*
      Create an ATM SkyStatus object for the observing altitude and frequency.
     */
    atm::AtmProfile atmo = get_atmprofile(altitude, temperature, pressure);
    atm::Frequency Freq(freq, "GHz");
    atm::RefractiveIndexProfile rip(Freq, atmo);

    return atm::SkyStatus(rip);
}


atm::SkyStatus get_sky_status_vec(double altitude, double temperature,
                                  double pressure,
                                  double freqmin, double freqmax,
                                  size_t nfreq) {
    /*
      Create an ATM SkyStatus object for the observing altitude and frequency.
     */
    atm::AtmProfile atmo = get_atmprofile(altitude, temperature, pressure);
    double freqstep = 0;
    if (nfreq > 1) freqstep = (freqmax - freqmin) / (nfreq - 1);
    // aatm SpectralGrid seems to have a bug.  The first grid point is
    // a whole grid step after the reference frequency.
    atm::SpectralGrid grid(nfreq, 0,
                           atm::Frequency(freqmin-freqstep, "GHz"),
                           atm::Frequency(freqstep, "GHz"));
    atm::RefractiveIndexProfile rip(grid, atmo);
    atm::SkyStatus ss(rip);
    return ss;
}


double toast::tatm::get_absorption_coefficient(double altitude,
                                               double temperature,
                                               double pressure,
                                               double pwv,
                                               double freq) {
    /*
      Return the dimensionless absorption coefficient for a zenith
      line of sight.

      Args:
          altitude : Observation altitude in meters.
          temperature : Observing temperature in Kelvins.
          pressure : Observing pressure in Pascals.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status(altitude, temperature, pressure, freq);
    ss.setUserWH2O(pwv, "mm");
    double opacity = ss.getWetOpacity().get();

    return 1 - exp(-opacity);
}


int toast::tatm::get_absorption_coefficient_vec(double altitude,
                                                double temperature,
                                                double pressure,
                                                double pwv,
                                                double freqmin, double freqmax,
                                                size_t nfreq,
                                                double *absorption) {
    /*
      Return the dimensionless absorption coefficient for a zenith
      line of sight.

      Args:
          altitude : Observation altitude in meters.
          temperature : Observing temperature in Kelvins.
          pressure : Observing pressure in Pascals.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status_vec(altitude, temperature, pressure,
                                           freqmin, freqmax, nfreq);
    ss.setUserWH2O(pwv, "mm");
    for (size_t i=0; i<nfreq; ++i) {
        double opacity = ss.getWetOpacity(i).get();
        absorption[i] = 1 - exp(-opacity);
    }

    return 0;
}


double toast::tatm::get_atmospheric_loading(double altitude,
                                            double temperature,
                                            double pressure,
                                            double pwv,
                                            double freq) {
    /*
      Return the equivalent black body temperature in Kelvin.

      Args:
          altitude : Observation altitude in meters.
          temperature : Observing temperature in Kelvins.
          pressure : Observing pressure in Pascals.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status(altitude, temperature, pressure, freq);
    ss.setUserWH2O(pwv, "mm");

    return ss.getTebbSky().get();
}

int toast::tatm::get_atmospheric_loading_vec(double altitude,
                                             double temperature,
                                             double pressure,
                                             double pwv,
                                             double freqmin, double freqmax,
                                             size_t nfreq, double *loading) {
    /*
      Return the equivalent black body temperature in Kelvin.

      Args:
          altitude : Observation altitude in meters.
          temperature : Observing temperature in Kelvins.
          pressure : Observing pressure in Pascals.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status_vec(altitude, temperature, pressure,
                                           freqmin, freqmax, nfreq);
    ss.setUserWH2O(pwv, "mm");
    for (unsigned int i=0; i<nfreq; ++i) {
        loading[i] = ss.getTebbSky(i).get();
    }

    return 0;
}
