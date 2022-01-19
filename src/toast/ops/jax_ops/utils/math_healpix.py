# see toast_math_healpix.cpp

"""
void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
                                    double * phi, int * region, double * z,
                                    double * rtz) const 
{
    toast::AlignedVector <double> work1(n);
    toast::AlignedVector <double> work2(n);
    toast::AlignedVector <double> work3(n);

    for (int64_t i = 0; i < n; ++i) 
    {
        int64_t offset = 3 * i;

        // region encodes BOTH the sign of Z and whether its
        // absolute value is greater than 2/3.

        z[i] = vec[offset + 2];

        double za = ::fabs(z[i]);

        int itemp = (z[i] > 0.0) ? 1 : -1;

        region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

        work1[i] = 3.0 * (1.0 - za);
        work3[i] = vec[offset + 1];
        work2[i] = vec[offset];
    }

    toast::vfast_sqrt(n, work1.data(), rtz);
    toast::vatan2(n, work3.data(), work2.data(), phi);

    return;
}

void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const 
{
    double eps = std::numeric_limits <float>::epsilon();
    
    for (int64_t i = 0; i < n; ++i) 
    {
        double ph = phi[i];
        if (fabs(ph) < eps) 
        {
            ph = 0.0;
        }
        double tt = (ph >= 0.0) ? ph * TWOINVPI : ph * TWOINVPI + 4.0;

        int64_t x;
        int64_t y;
        double temp1;
        double temp2;
        int64_t jp;
        int64_t jm;
        int64_t ifp;
        int64_t ifm;
        int64_t face;
        int64_t ntt;
        double tp;

        if (::abs(region[i]) == 1) 
        {
            temp1 = halfnside_ + dnside_ * tt;
            temp2 = tqnside_ * z[i];

            jp = static_cast <int64_t> (temp1 - temp2);
            jm = static_cast <int64_t> (temp1 + temp2);

            ifp = jp >> factor_;
            ifm = jm >> factor_;

            face;
            if (ifp == ifm) 
            {
                face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
            } 
            else if (ifp < ifm) 
            {
                face = ifp;
            } 
            else 
            {
                face = ifm + 8;
            }

            x = jm & nsideminusone_;
            y = nsideminusone_ - (jp & nsideminusone_);
        } 
        else 
        {
            ntt = static_cast <int64_t> (tt);

            tp = tt - static_cast <double> (ntt);

            temp1 = dnside_ * rtz[i];

            jp = static_cast <int64_t> (tp * temp1);
            jm = static_cast <int64_t> ((1.0 - tp) * temp1);

            if (jp >= nside_) 
            {
                jp = nsideminusone_;
            }
            if (jm >= nside_) 
            {
                jm = nsideminusone_;
            }

            if (z[i] >= 0) 
            {
                face = ntt;
                x = nsideminusone_ - jm;
                y = nsideminusone_ - jp;
            } 
            else 
            {
                face = ntt + 8;
                x = jp;
                y = jm;
            }
        }

        uint64_t sipf = xy2pix_(static_cast <uint64_t> (x), static_cast <uint64_t> (y));
        pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
    }
}

void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
                                    int64_t * pix) const 
{
    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
    zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
}
"""