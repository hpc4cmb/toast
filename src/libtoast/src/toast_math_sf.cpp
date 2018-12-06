
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast/utils.hpp>
#include <toast/sf.hpp>

#ifdef HAVE_MKL
# include <mkl.h>
#endif // ifdef HAVE_MKL

#include <cmath>
#include <vector>


#ifdef HAVE_MKL

// These call MKL VM functions with "High Accuracy" mode.

void toast::vsin(int n, double const * ang, double * sinout) {
    vmdSin(n, ang, sinout, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vcos(int n, double const * ang, double * cosout) {
    vmdCos(n, ang, cosout, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vsincos(int n, double const * ang, double * sinout,
                    double * cosout) {
    vmdSinCos(n, ang, sinout, cosout,
              VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vatan2(int n, double const * y, double const * x,
                   double * ang) {
    vmdAtan2(n, y, x, ang, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vsqrt(int n, double const * in, double * out) {
    vmdSqrt(n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vrsqrt(int n, double const * in, double * out) {
    vmdInvSqrt(n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vexp(int n, double const * in, double * out) {
    vmdExp(n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vlog(int n, double const * in, double * out) {
    vmdLn(n, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

// These call MKL VM functions with "Low Accuracy" mode.

void toast::vfast_sin(int n, double const * ang, double * sinout) {
    vmdSin(n, ang, sinout, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_cos(int n, double const * ang, double * cosout) {
    vmdCos(n, ang, cosout, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_sincos(int n, double const * ang, double * sinout,
                         double * cosout) {
    vmdSinCos(n, ang, sinout, cosout,
              VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_atan2(int n, double const * y, double const * x,
                        double * ang) {
    vmdAtan2(n, y, x, ang, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_sqrt(int n, double const * in, double * out) {
    vmdSqrt(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_rsqrt(int n, double const * in, double * out) {
    vmdInvSqrt(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_exp(int n, double const * in, double * out) {
    vmdExp(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_log(int n, double const * in, double * out) {
    vmdLn(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_erfinv(int n, double const * in, double * out) {
    vmdErfInv(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

#else // ifdef HAVE_MKL


// These are simply threaded for-loops that call the standard
// math library functions.

void toast::vsin(int n, double const * ang, double * sinout) {
    if (toast::is_aligned(ang) && toast::is_aligned(sinout)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            sinout[i] = ::sin(ang[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            sinout[i] = ::sin(ang[i]);
        }
    }
    return;
}

void toast::vcos(int n, double const * ang, double * cosout) {
    if (toast::is_aligned(ang) && toast::is_aligned(cosout)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            cosout[i] = ::cos(ang[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            cosout[i] = ::cos(ang[i]);
        }
    }
    return;
}

void toast::vsincos(int n, double const * ang, double * sinout,
                    double * cosout) {
    if (toast::is_aligned(ang) && toast::is_aligned(sinout) &&
        toast::is_aligned(cosout)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            sinout[i] = ::sin(ang[i]);
            cosout[i] = ::cos(ang[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            sinout[i] = ::sin(ang[i]);
            cosout[i] = ::cos(ang[i]);
        }
    }
    return;
}

void toast::vatan2(int n, double const * y, double const * x,
                   double * ang) {
    if (toast::is_aligned(ang) && toast::is_aligned(x) &&
        toast::is_aligned(y)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            ang[i] = ::atan2(y[i], x[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            ang[i] = ::atan2(y[i], x[i]);
        }
    }
    return;
}

void toast::vsqrt(int n, double const * in, double * out) {
    if (toast::is_aligned(in) && toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = ::sqrt(in[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = ::sqrt(in[i]);
        }
    }
    return;
}

void toast::vrsqrt(int n, double const * in, double * out) {
    if (toast::is_aligned(in) && toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = 1.0 / ::sqrt(in[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = 1.0 / ::sqrt(in[i]);
        }
    }
    return;
}

void toast::vexp(int n, double const * in, double * out) {
    if (toast::is_aligned(in) && toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = ::exp(in[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = ::exp(in[i]);
        }
    }
    return;
}

void toast::vlog(int n, double const * in, double * out) {
    if (toast::is_aligned(in) && toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = ::log(in[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = ::log(in[i]);
        }
    }
    return;
}

// These use polynomial approximations for some functions.

void toast::vfast_sin(int n, double const * ang, double * sinout) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double sx;
    double ssign;
    double sx2;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    # pragma \
    omp parallel for default(shared) private(i, sx, sx2, ssign, quot, rem, x, quad) schedule(static)
    for (i = 0; i < n; i++) {
        quot = ang[i] * toast::INV_TWOPI;
        rem = quot - floor(quot);
        x = rem * toast::TWOPI;
        while (x < 0.0) {
            x += toast::TWOPI;
        }
        quad = static_cast <int> (x * toast::TWOINVPI);

        switch (quad) {
            case 1:
                sx = x - toast::PI_2;
                ssign = 1.0;
                break;

            case 2:
                sx = toast::THREEPI_2 - x;
                ssign = -1.0;
                break;

            case 3:
                sx = x - toast::THREEPI_2;
                ssign = -1.0;
                break;

            default:
                sx = toast::PI_2 - x;
                ssign = 1.0;
                break;
        }

        sx2 = sx * sx;
        sinout[i] = ssign * (SC1 + sx2 *
                             (SC2 + sx2 *
                              (SC3 + sx2 *
                               (SC4 + sx2 *
                                (SC5 + sx2 *
                                 (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
    }
    return;
}

void toast::vfast_cos(int n, double const * ang, double * cosout) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double cx;
    double cx2;
    double csign;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    # pragma \
    omp parallel for default(shared) private(i, cx, cx2, csign, quot, rem, x, quad) schedule(static)
    for (i = 0; i < n; i++) {
        quot = ang[i] * toast::INV_TWOPI;
        rem = quot - floor(quot);
        x = rem * toast::TWOPI;
        while (x < 0.0) {
            x += toast::TWOPI;
        }
        quad = static_cast <int> (x * toast::TWOINVPI);
        switch (quad) {
            case 1:
                cx = toast::PI - x;
                csign = -1.0;
                break;

            case 2:
                cx = x - toast::PI;
                csign = -1.0;
                break;

            case 3:
                cx = toast::TWOPI - x;
                csign = 1.0;
                break;

            default:
                cx = x;
                csign = 1.0;
                break;
        }
        cx2 = cx * cx;
        cosout[i] = csign * (SC1 + cx2 *
                             (SC2 + cx2 *
                              (SC3 + cx2 *
                               (SC4 + cx2 *
                                (SC5 + cx2 *
                                 (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
    }
    return;
}

void toast::vfast_sincos(int n, double const * ang, double * sinout,
                         double * cosout) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double sx;
    double cx;
    double sx2;
    double cx2;
    double ssign;
    double csign;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    # pragma \
    omp parallel for default(shared) private(i, sx, cx, sx2, cx2, ssign, csign, quot, rem, x, quad) schedule(static)
    for (i = 0; i < n; i++) {
        quot = ang[i] * toast::INV_TWOPI;
        rem = quot - floor(quot);
        x = rem * toast::TWOPI;
        while (x < 0.0) {
            x += toast::TWOPI;
        }
        quad = static_cast <int> (x * toast::TWOINVPI);
        switch (quad) {
            case 1:
                sx = x - toast::PI_2;
                ssign = 1.0;
                cx = toast::PI - x;
                csign = -1.0;
                break;

            case 2:
                sx = toast::THREEPI_2 - x;
                ssign = -1.0;
                cx = x - toast::PI;
                csign = -1.0;
                break;

            case 3:
                sx = x - toast::THREEPI_2;
                ssign = -1.0;
                cx = toast::TWOPI - x;
                csign = 1.0;
                break;

            default:
                sx = toast::PI_2 - x;
                ssign = 1.0;
                cx = x;
                csign = 1.0;
                break;
        }
        sx2 = sx * sx;
        cx2 = cx * cx;

        sinout[i] = ssign * (SC1 + sx2 *
                             (SC2 + sx2 *
                              (SC3 + sx2 *
                               (SC4 + sx2 *
                                (SC5 + sx2 *
                                 (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
        cosout[i] = csign * (SC1 + cx2 *
                             (SC2 + cx2 *
                              (SC3 + cx2 *
                               (SC4 + cx2 *
                                (SC5 + cx2 *
                                 (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
    }
    return;
}

void toast::vfast_atan2(int n, double const * y, double const * x,
                        double * ang) {
    double const ATCHEB1 = 48.70107004404898384;
    double const ATCHEB2 = 49.5326263772254345;
    double const ATCHEB3 = 9.40604244231624;
    double const ATCHEB4 = 48.70107004404996166;
    double const ATCHEB5 = 65.7663163908956299;
    double const ATCHEB6 = 21.587934067020262;
    int i;
    double r2;
    double r;
    int complement;
    int region;
    int sign;

    # pragma \
    omp parallel for default(shared) private(i, r, r2, complement, region, sign) schedule(static)
    for (i = 0; i < n; i++) {
        r = y[i] / x[i];

        // reduce range to PI/12

        complement = 0;
        region = 0;
        sign = 0;
        if (r < 0) {
            r = -r;
            sign = 1;
        }
        if (r > 1.0) {
            r = 1.0 / r;
            complement = 1;
        }
        if (r > toast::TANTWELFTHPI) {
            r = (r - toast::TANSIXTHPI) / (1 + toast::TANSIXTHPI * r);
            region = 1;
        }
        r2 = r * r;
        r = (r * (ATCHEB1 + r2 * (ATCHEB2 + r2 * ATCHEB3))) /
            (ATCHEB4 + r2 * (ATCHEB5 + r2 * (ATCHEB6 + r2)));
        if (region) {
            r += toast::SIXTHPI;
        }
        if (complement) {
            r = toast::PI_2 - r;
        }
        if (sign) {
            r = -r;
        }

        // adjust quadrant

        if (x[i] > 0.0) {
            ang[i] = r;
        } else if (x[i] < 0.0) {
            ang[i] = r + toast::PI;
            if (ang[i] > toast::PI) {
                ang[i] -= toast::TWOPI;
            }
        } else if (y[i] > 0.0) {
            ang[i] = toast::PI_2;
        } else if (y[i] < 0.0) {
            ang[i] = -toast::PI_2;
        } else {
            ang[i] = 0.0;
        }
    }
    return;
}

void toast::vfast_sqrt(int n, double const * in, double * out) {
    toast::vsqrt(n, in, out);
    return;
}

void toast::vfast_rsqrt(int n, double const * in, double * out) {
    toast::vrsqrt(n, in, out);
    return;
}

void toast::vfast_exp(int n, double const * in, double * out) {
    toast::vexp(n, in, out);
    return;
}

void toast::vfast_log(int n, double const * in, double * out) {
    toast::vlog(n, in, out);
    return;
}

void toast::vfast_erfinv(int n, double const * in, double * out) {
    // Based on domain decomposition by Mike Giles here:
    // https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
    //
    // With numerical constants obtained from:
    // https://people.maths.ox.ac.uk/gilesm/codes/erfinv/
    //
    std::vector <double, toast::simd_allocator <double> > arg(n);
    std::vector <double, toast::simd_allocator <double> > lg(n);

    if (toast::is_aligned(in)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            double ab = ::fabs(in[i]);
            arg[i] = (1.0 - ab) * (1.0 + ab);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            double ab = ::fabs(in[i]);
            arg[i] = (1.0 - ab) * (1.0 + ab);
        }
    }

    toast::vfast_log(n, arg.data(), lg.data());

    if (toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            double w = -lg[i];
            double p;

            if (w < 6.250000) {
                w = w - 3.125000;
                p =  -3.6444120640178196996e-21;
                p *= w;
                p += -1.685059138182016589e-19;
                p *= w;
                p += 1.2858480715256400167e-18;
                p *= w;
                p += 1.115787767802518096e-17;
                p *= w;
                p += -1.333171662854620906e-16;
                p *= w;
                p += 2.0972767875968561637e-17;
                p *= w;
                p += 6.6376381343583238325e-15;
                p *= w;
                p += -4.0545662729752068639e-14;
                p *= w;
                p += -8.1519341976054721522e-14;
                p *= w;
                p += 2.6335093153082322977e-12;
                p *= w;
                p += -1.2975133253453532498e-11;
                p *= w;
                p += -5.4154120542946279317e-11;
                p *= w;
                p += 1.051212273321532285e-09;
                p *= w;
                p += -4.1126339803469836976e-09;
                p *= w;
                p += -2.9070369957882005086e-08;
                p *= w;
                p += 4.2347877827932403518e-07;
                p *= w;
                p += -1.3654692000834678645e-06;
                p *= w;
                p += -1.3882523362786468719e-05;
                p *= w;
                p += 0.0001867342080340571352;
                p *= w;
                p += -0.00074070253416626697512;
                p *= w;
                p += -0.0060336708714301490533;
                p *= w;
                p += 0.24015818242558961693;
                p *= w;
                p += 1.6536545626831027356;
            } else if (w < 16.000000) {
                w = ::sqrt(w) - 3.250000;
                p = 2.2137376921775787049e-09;
                p *= w;
                p += 9.0756561938885390979e-08;
                p *= w;
                p += -2.7517406297064545428e-07;
                p *= w;
                p += 1.8239629214389227755e-08;
                p *= w;
                p += 1.5027403968909827627e-06;
                p *= w;
                p += -4.013867526981545969e-06;
                p *= w;
                p += 2.9234449089955446044e-06;
                p *= w;
                p += 1.2475304481671778723e-05;
                p *= w;
                p += -4.7318229009055733981e-05;
                p *= w;
                p += 6.8284851459573175448e-05;
                p *= w;
                p += 2.4031110387097893999e-05;
                p *= w;
                p += -0.0003550375203628474796;
                p *= w;
                p += 0.00095328937973738049703;
                p *= w;
                p += -0.0016882755560235047313;
                p *= w;
                p += 0.0024914420961078508066;
                p *= w;
                p += -0.0037512085075692412107;
                p *= w;
                p += 0.005370914553590063617;
                p *= w;
                p += 1.0052589676941592334;
                p *= w;
                p += 3.0838856104922207635;
            } else {
                w = ::sqrt(w) - 5.000000;
                p = -2.7109920616438573243e-11;
                p *= w;
                p += -2.5556418169965252055e-10;
                p *= w;
                p += 1.5076572693500548083e-09;
                p *= w;
                p += -2.5556418169965252055e-10;
                p *= w;
                p += 1.5076572693500548083e-09;
                p *= w;
                p += -3.7894654401267369937e-09;
                p *= w;
                p += 7.6157012080783393804e-09;
                p *= w;
                p += -1.4960026627149240478e-08;
                p *= w;
                p += 2.9147953450901080826e-08;
                p *= w;
                p += -6.7711997758452339498e-08;
                p *= w;
                p += 2.2900482228026654717e-07;
                p *= w;
                p += -6.7711997758452339498e-08;
                p *= w;
                p += 2.2900482228026654717e-07;
                p *= w;
                p += -9.9298272942317002539e-07;
                p *= w;
                p += 4.5260625972231537039e-06;
                p *= w;
                p += -1.9681778105531670567e-05;
                p *= w;
                p += 7.5995277030017761139e-05;
                p *= w;
                p += -0.00021503011930044477347;
                p *= w;
                p += -0.00013871931833623122026;
                p *= w;
                p += 1.0103004648645343977;
                p *= w;
                p += 4.8499064014085844221;
            }

            out[i] = p * in[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            double w = -lg[i];
            double p;

            if (w < 6.250000) {
                w = w - 3.125000;
                p =  -3.6444120640178196996e-21;
                p *= w;
                p += -1.685059138182016589e-19;
                p *= w;
                p += 1.2858480715256400167e-18;
                p *= w;
                p += 1.115787767802518096e-17;
                p *= w;
                p += -1.333171662854620906e-16;
                p *= w;
                p += 2.0972767875968561637e-17;
                p *= w;
                p += 6.6376381343583238325e-15;
                p *= w;
                p += -4.0545662729752068639e-14;
                p *= w;
                p += -8.1519341976054721522e-14;
                p *= w;
                p += 2.6335093153082322977e-12;
                p *= w;
                p += -1.2975133253453532498e-11;
                p *= w;
                p += -5.4154120542946279317e-11;
                p *= w;
                p += 1.051212273321532285e-09;
                p *= w;
                p += -4.1126339803469836976e-09;
                p *= w;
                p += -2.9070369957882005086e-08;
                p *= w;
                p += 4.2347877827932403518e-07;
                p *= w;
                p += -1.3654692000834678645e-06;
                p *= w;
                p += -1.3882523362786468719e-05;
                p *= w;
                p += 0.0001867342080340571352;
                p *= w;
                p += -0.00074070253416626697512;
                p *= w;
                p += -0.0060336708714301490533;
                p *= w;
                p += 0.24015818242558961693;
                p *= w;
                p += 1.6536545626831027356;
            } else if (w < 16.000000) {
                w = ::sqrt(w) - 3.250000;
                p = 2.2137376921775787049e-09;
                p *= w;
                p += 9.0756561938885390979e-08;
                p *= w;
                p += -2.7517406297064545428e-07;
                p *= w;
                p += 1.8239629214389227755e-08;
                p *= w;
                p += 1.5027403968909827627e-06;
                p *= w;
                p += -4.013867526981545969e-06;
                p *= w;
                p += 2.9234449089955446044e-06;
                p *= w;
                p += 1.2475304481671778723e-05;
                p *= w;
                p += -4.7318229009055733981e-05;
                p *= w;
                p += 6.8284851459573175448e-05;
                p *= w;
                p += 2.4031110387097893999e-05;
                p *= w;
                p += -0.0003550375203628474796;
                p *= w;
                p += 0.00095328937973738049703;
                p *= w;
                p += -0.0016882755560235047313;
                p *= w;
                p += 0.0024914420961078508066;
                p *= w;
                p += -0.0037512085075692412107;
                p *= w;
                p += 0.005370914553590063617;
                p *= w;
                p += 1.0052589676941592334;
                p *= w;
                p += 3.0838856104922207635;
            } else {
                w = ::sqrt(w) - 5.000000;
                p = -2.7109920616438573243e-11;
                p *= w;
                p += -2.5556418169965252055e-10;
                p *= w;
                p += 1.5076572693500548083e-09;
                p *= w;
                p += -2.5556418169965252055e-10;
                p *= w;
                p += 1.5076572693500548083e-09;
                p *= w;
                p += -3.7894654401267369937e-09;
                p *= w;
                p += 7.6157012080783393804e-09;
                p *= w;
                p += -1.4960026627149240478e-08;
                p *= w;
                p += 2.9147953450901080826e-08;
                p *= w;
                p += -6.7711997758452339498e-08;
                p *= w;
                p += 2.2900482228026654717e-07;
                p *= w;
                p += -6.7711997758452339498e-08;
                p *= w;
                p += 2.2900482228026654717e-07;
                p *= w;
                p += -9.9298272942317002539e-07;
                p *= w;
                p += 4.5260625972231537039e-06;
                p *= w;
                p += -1.9681778105531670567e-05;
                p *= w;
                p += 7.5995277030017761139e-05;
                p *= w;
                p += -0.00021503011930044477347;
                p *= w;
                p += -0.00013871931833623122026;
                p *= w;
                p += 1.0103004648645343977;
                p *= w;
                p += 4.8499064014085844221;
            }

            out[i] = p * in[i];
        }
    }

    return;
}

#endif // ifdef HAVE_MKL
