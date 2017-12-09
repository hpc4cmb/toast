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


atm::Pressure get_pressure(atm::Length Alt, atm::Pressure P0,
                           atm::Temperature T) {
    /*
      Return an approximate atmospheric pressure at the observing
      altitude.
      
      Args:
          altitude : Altitude in meters.
          P0 : Sea level pressure.
          T : Temperature at the observing site.
     */

    const double R = 8.3144598; // Ideal gas constant [J / mol / K]
    const double Ma = 29e-3; // Air molecular weight [kg/mol]
    const double g = 9.80665; // gravitational acceleration [m/s^2]
    double atm_scale_height = R * T.get("K") / Ma / g;

    return P0 * exp(-Alt.get("m") / atm_scale_height);
}


atm::SkyStatus get_sky_status(double altitude, double freq) {
    /*
      Create an ATM SkyStatus object for the observing altitude and frequency.
     */

    unsigned int atmType = 1; // Atmospheric type (to reproduce
                              // behavior above the tropopause)
    atm::Temperature T(270, "K"); // Ground level temperature (K)
    atm::Length Alt(altitude, "m"); // Altitude of the site (m)
    atm::Length WVL(2, "km"); // Water vapour scale height (km)
    double TLR=-5.6; // Tropospheric lapse rate (K/km)
    atm::Length topAtm(48.0, "km"); // Upper atm. boundary for calculations
    atm::Pressure Pstep(1.0, "mb"); // Primary pressure step
    double PstepFact=1.2; // Pressure step ratio between two
                          // consecutive layers
    atm::Pressure P0(1000, "mb"); // Ocean level pressure (mBar)
    atm::Pressure P = get_pressure(Alt, P0, T);
    atm::Humidity H(10, "%");

    atm::AtmProfile atmo(Alt, P, T, TLR, H, WVL, Pstep, PstepFact,
                         topAtm, atmType);

    atm::Frequency Freq(freq, "GHz");
    atm::RefractiveIndexProfile rip(Freq, atmo);
    atm::SkyStatus ss(rip);
    return ss;
}


double toast::tatm::get_absorption_coefficient(double altitude, double pwv,
                                               double freq) {
    /*
      Return the dimensionless absorption coefficient for a zenith
      line of sight.
      
      Args:
          altitude : Observation altitude in meters.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status(altitude, freq);
    ss.setUserWH2O(pwv, "mm");
    double opacity = ss.getWetOpacity().get();

    return 1 - exp(-opacity);
}


double toast::tatm::get_atmospheric_loading(double altitude, double pwv,
                                           double freq) {
    /*
      Return the equivalent black body temperature in Kelvin.

      Args:
          altitude : Observation altitude in meters.
          pwv : Precipitable water vapor column height in mm.
          freq : Observing frequency in GHz.
    */
    atm::SkyStatus ss = get_sky_status(altitude, freq);
    ss.setUserWH2O(pwv, "mm");
    double opacity = ss.getWetOpacity().get();

    return ss.getTebbSky().get();
}
