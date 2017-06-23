/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/

#ifndef TOAST_FFT_HPP
#define TOAST_FFT_HPP


namespace toast { namespace fft {

    enum class plan_type {
        fast,
        best
    };
  
    enum class direction {
        forward,
        backward
    };
  

    // This uses aligned memory allocation
    typedef std::vector < double, toast::mem::simd_allocator < double > > fft_data;

  
    class r1d {

        public :

            static r1d * create ( int64_t length, int64_t n, plan_type type, direction dir, double scale );

            virtual ~r1d ( ) { }          

            virtual void exec ( ) { return; }

            virtual std::vector < double * > tdata ( ) { return std::vector < double * > (); }
            virtual std::vector < double * > fdata ( ) { return std::vector < double * > (); }

            int64_t length ( );

            int64_t count ( );

        protected :
            
            r1d ( int64_t length, int64_t n, plan_type type, direction dir, double scale );

            int64_t length_;
            int64_t n_;
            double scale_;
            plan_type type_;
            direction dir_;

    };

    typedef std::shared_ptr < r1d > r1d_p;


    // R1D FFT plan store

    class r1d_plan_store {

        public:
            ~r1d_plan_store ( );
            static r1d_plan_store & get ( );
            void cache ( int64_t len, int64_t n = 1 );
            r1d_p forward ( int64_t len, int64_t n = 1 );
            r1d_p backward ( int64_t len, int64_t n = 1 );
            void clear ( );

        private:
            r1d_plan_store ( ) { }
            std::map < int, std::map < std::pair < int64_t, int64_t >, r1d_p > > fplans_;
            std::map < int, std::map < std::pair < int64_t, int64_t >, r1d_p > > rplans_;

    };



} }

#endif

