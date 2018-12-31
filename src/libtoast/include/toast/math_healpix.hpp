
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#ifndef TOAST_MATH_HEALPIX_HPP
#define TOAST_MATH_HEALPIX_HPP


namespace toast {

void healpix_ang2vec(int64_t n, double const * theta, double const * phi,
                     double * vec);

void healpix_vec2ang(int64_t n, double const * vec, double * theta,
                     double * phi);

void healpix_vecs2angpa(int64_t n, double const * vec, double * theta,
                        double * phi, double * pa);

class HealpixPixels {
    public:

        HealpixPixels();
        HealpixPixels(int64_t nside);
        ~HealpixPixels() {}

        void reset(int64_t nside);

        void vec2zphi(int64_t n, double const * vec, double * phi,
                      int * region, double * z, double * rtz) const;

        void theta2z(int64_t n, double const * theta, int * region, double * z,
                     double * rtz) const;

        void zphi2nest(int64_t n, double const * phi, int const * region,
                       double const * z, double const * rtz,
                       int64_t * pix) const;

        void zphi2ring(int64_t n, double const * phi, int const * region,
                       double const * z, double const * rtz,
                       int64_t * pix) const;

        void ang2nest(int64_t n, double const * theta, double const * phi,
                      int64_t * pix) const;

        void ang2ring(int64_t n, double const * theta, double const * phi,
                      int64_t * pix) const;

        void vec2nest(int64_t n, double const * vec, int64_t * pix) const;

        void vec2ring(int64_t n, double const * vec, int64_t * pix) const;

        void ring2nest(int64_t n, int64_t const * ringpix,
                       int64_t * nestpix) const;

        void nest2ring(int64_t n, int64_t const * nestpix,
                       int64_t * ringpix) const;

        void degrade_ring(int factor, int64_t n, int64_t const * inpix,
                          int64_t * outpix) const;

        void degrade_nest(int factor, int64_t n, int64_t const * inpix,
                          int64_t * outpix) const;

        void upgrade_ring(int factor, int64_t n, int64_t const * inpix,
                          int64_t * outpix) const;

        void upgrade_nest(int factor, int64_t n, int64_t const * inpix,
                          int64_t * outpix) const;

    private:

        void init();

        uint64_t xy2pix_(uint64_t x, uint64_t y) const {
            return utab_[x & 0xff] | (utab_[(x >> 8) & 0xff] << 16) |
                   (utab_[(x >> 16) & 0xff] << 32) |
                   (utab_[(x >> 24) & 0xff] << 48) |
                   (utab_[y & 0xff] << 1) | (utab_[(y >> 8) & 0xff] << 17) |
                   (utab_[(y >> 16) & 0xff] << 33) |
                   (utab_[(y >> 24) & 0xff] << 49);
        }

        uint64_t x2pix_(uint64_t x) const {
            return utab_[x & 0xff] | (utab_[x >> 8] << 16) |
                   (utab_[(x >> 16) & 0xff] << 32) |
                   (utab_[(x >> 24) & 0xff] << 48);
        }

        uint64_t y2pix_(uint64_t y) const {
            return (utab_[y & 0xff] << 1) | (utab_[y >> 8] << 17) |
                   (utab_[(y >> 16) & 0xff] << 33) |
                   (utab_[(y >> 24) & 0xff] << 49);
        }

        void pix2xy_(uint64_t pix, uint64_t & x, uint64_t & y) const {
            uint64_t raw;
            raw = (pix & 0x5555ull) | ((pix & 0x55550000ull) >> 15) |
                  ((pix & 0x555500000000ull) >> 16) |
                  ((pix & 0x5555000000000000ull) >> 31);
            x = ctab_[raw & 0xff] | (ctab_[(raw >> 8) & 0xff] << 4) |
                (ctab_[(raw >> 16) & 0xff] << 16) |
                (ctab_[(raw >> 24) & 0xff] << 20);
            raw = ((pix & 0xaaaaull) >> 1) | ((pix & 0xaaaa0000ull) >> 16) |
                  ((pix & 0xaaaa00000000ull) >> 17) |
                  ((pix & 0xaaaa000000000000ull) >> 32);
            y = ctab_[raw & 0xff] | (ctab_[(raw >> 8) & 0xff] << 4) |
                (ctab_[(raw >> 16) & 0xff] << 16) |
                (ctab_[(raw >> 24) & 0xff] << 20);
            return;
        }

        static const int64_t jr_[];
        static const int64_t jp_[];
        uint64_t utab_[0x100];
        uint64_t ctab_[0x100];
        int64_t nside_;
        int64_t npix_;
        int64_t ncap_;
        double dnside_;
        int64_t twonside_;
        int64_t fournside_;
        int64_t nsideplusone_;
        int64_t nsideminusone_;
        double halfnside_;
        double tqnside_;
        int64_t factor_;
};

}

#endif // ifndef TOAST_HEALPIX_HPP
