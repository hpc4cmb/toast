/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/

#include <ctoast.hpp>

#include <sstream>
#include <cstring>

typedef toast::util::timing_manager                 timing_manager;
typedef toast::util::timing_manager::toast_timer_t  toast_timer_t;
typedef toast::util::details::base_timer            base_timer_t;

//============================================================================//
// Global library initialize / finalize
//============================================================================//

void ctoast_init ( int argc, char *argv[] )
{
    toast::init ( argc, argv );
    return;
}

void ctoast_finalize ( )
{
    toast::finalize ( );
    return;
}

void ctoast_raise_error(int errcode)
{
    std::stringstream serr;
    toast::TerminationSignalMessage(errcode, nullptr, serr);
    std::cerr << serr.str() << std::endl;
}

//============================================================================//
// Util sub-library
//============================================================================//

int ctoast_timers_enabled()
{
#if defined(DISABLE_TIMERS)
    return 0;
#else
    return timing_manager::is_enabled();
#endif
}

//----------------------------------------------------------------------------//

void ctoast_timers_toggle(int32_t val)
{
    timing_manager::instance()->enable(val);
}

//----------------------------------------------------------------------------//
ctoast_timer* ctoast_get_simple_timer(char* ckey, char* _cfmt)
{
#if defined(DISABLE_TIMERS)
    return nullptr;
#endif
    std::string cfmt = const_cast<const char*>(_cfmt);
    std::string fmt = (cfmt.empty())
                      ? toast::util::timer::default_format
                      : cfmt;
    toast_timer_t* _t = new toast_timer_t(ckey, "", fmt);
    return reinterpret_cast<ctoast_timer*>(_t);
}

//----------------------------------------------------------------------------//

void ctoast_del_simple_timer(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return;
#endif
    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    delete _t;
}

//----------------------------------------------------------------------------//

ctoast_timer* ctoast_get_timer(char* ckey)
{
#if defined(DISABLE_TIMERS)
    return nullptr;
#endif

    toast_timer_t& _t = timing_manager::instance()
                        ->timer(ckey, "pyc", base_timer_t::get_instance_count(),
                                base_timer_t::get_instance_hash());
    return reinterpret_cast<ctoast_timer*>(&_t);
}

//----------------------------------------------------------------------------//

ctoast_timer* ctoast_get_timer_at(int32_t i)
{
#if defined(DISABLE_TIMERS)
    return nullptr;
#endif

    toast_timer_t& _t = timing_manager::instance()->at(i);
    return reinterpret_cast<ctoast_timer*>(&_t);
}

//----------------------------------------------------------------------------//

void ctoast_timer_start(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->start();
}

//----------------------------------------------------------------------------//

void ctoast_timer_stop(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->stop();
}

//----------------------------------------------------------------------------//

void ctoast_timer_report(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->report();
}

//----------------------------------------------------------------------------//

uint64_t ctoast_get_timer_instance_count()
{
#if defined(DISABLE_TIMERS)
    return 0;
#endif

    return toast::util::details::base_timer::get_instance_count();
}

//----------------------------------------------------------------------------//

void ctoast_op_timer_instance_count(int32_t op, int32_t nhash)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    int64_t _new = op + toast::util::details::base_timer::get_instance_count();
    if(!(_new < 0))
        toast::util::details::base_timer::get_instance_count() += op;

    toast::util::details::base_timer::get_instance_hash() += nhash;
}

//----------------------------------------------------------------------------//

ctoast_timing_manager* ctoast_get_timing_manager()
{
#if defined(DISABLE_TIMERS)
    return nullptr;
#endif

    return reinterpret_cast<ctoast_timing_manager*>(timing_manager::instance());
}

//----------------------------------------------------------------------------//

void ctoast_set_timing_output_file(char* cfname)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    std::string fname = std::string(const_cast<const char*>(cfname));
    timing_manager::instance()->set_output_stream(fname);
}

//----------------------------------------------------------------------------//

void ctoast_serialize_timing_manager(char* cfname)
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    std::string fname = std::string(const_cast<const char*>(cfname));
    timing_manager::instance()->write_json(fname);
}

//----------------------------------------------------------------------------//

void ctoast_timing_manager_report()
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    timing_manager::instance()->report();
}

//----------------------------------------------------------------------------//

void ctoast_timing_manager_clear()
{
#if defined(DISABLE_TIMERS)
    return;
#endif

    timing_manager::instance()->clear();
}

//----------------------------------------------------------------------------//

void ctoast_timing_manager_set_max_depth(int32_t _depth)
{
#if defined(DISABLE_TIMERS)
    return;
#endif
    timing_manager::max_depth() = _depth;
}

//----------------------------------------------------------------------------//

int32_t ctoast_timing_manager_max_depth()
{
#if defined(DISABLE_TIMERS)
    return 0;
#endif
    return timing_manager::max_depth();
}

//----------------------------------------------------------------------------//

size_t ctoast_timing_manager_size()
{
#if defined(DISABLE_TIMERS)
    return 0;
#endif

    return timing_manager::instance()->size();
}

//----------------------------------------------------------------------------//

double ctoast_timer_real_elapsed(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return 0.0;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->stop();
    return _t->real_elapsed();
}

//----------------------------------------------------------------------------//

double ctoast_timer_system_elapsed(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return 0.0;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->stop();
    return _t->system_elapsed();
}

//----------------------------------------------------------------------------//

double ctoast_timer_user_elapsed(ctoast_timer* _ct)
{
#if defined(DISABLE_TIMERS)
    return 0.0;
#endif

    toast_timer_t* _t = reinterpret_cast<toast_timer_t*>(_ct);
    _t->stop();
    return _t->user_elapsed();
}

//============================================================================//
// Math sub-library
//============================================================================//

// aligned memory

void * ctoast_mem_aligned_alloc ( size_t size ) {
    void * data = toast::mem::aligned_alloc ( size, toast::mem::SIMD_ALIGN );
    return data;
}

void ctoast_mem_aligned_free ( void * data ) {
    toast::mem::aligned_free ( data );
    return;
}

// special functions

void ctoast_sf_sin ( int n, double const * ang, double * sinout ) {
    TOAST_AUTO_TIMER();
    toast::sf::sin( n, ang, sinout );
    return;
}

void ctoast_sf_cos ( int n, double const * ang, double * cosout ) {
    TOAST_AUTO_TIMER();
    toast::sf::cos( n, ang, cosout );
    return;
}

void ctoast_sf_sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    TOAST_AUTO_TIMER();
    toast::sf::sincos( n, ang, sinout, cosout );
    return;
}

void ctoast_sf_atan2 ( int n, double const * y, double const * x, double * ang ) {
    TOAST_AUTO_TIMER();
    toast::sf::atan2( n, y, x, ang );
    return;
}

void ctoast_sf_sqrt ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::sqrt( n, in, out );
    return;
}

void ctoast_sf_rsqrt ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::rsqrt( n, in, out );
    return;
}

void ctoast_sf_exp ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::exp( n, in, out );
    return;
}

void ctoast_sf_log ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::log( n, in, out );
    return;
}

void ctoast_sf_fast_sin ( int n, double const * ang, double * sinout ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_sin( n, ang, sinout );
    return;
}

void ctoast_sf_fast_cos ( int n, double const * ang, double * cosout ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_cos( n, ang, cosout );
    return;
}

void ctoast_sf_fast_sincos ( int n, double const * ang, double * sinout, double * cosout ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_sincos( n, ang, sinout, cosout );
    return;
}

void ctoast_sf_fast_atan2 ( int n, double const * y, double const * x, double * ang ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_atan2( n, y, x, ang );
    return;
}

void ctoast_sf_fast_sqrt ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_sqrt( n, in, out );
    return;
}

void ctoast_sf_fast_rsqrt ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_rsqrt( n, in, out );
    return;
}

void ctoast_sf_fast_exp ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_exp( n, in, out );
    return;
}

void ctoast_sf_fast_log ( int n, double const * in, double * out ) {
    TOAST_AUTO_TIMER();
    toast::sf::fast_log( n, in, out );
    return;
}

//----------------------------------------------------------------------------//
// RNG

void ctoast_rng_dist_uint64 ( size_t n, uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              uint64_t* data )
{
    TOAST_AUTO_TIMER();
    toast::rng::dist_uint64(n, key1, key2, counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_01 ( size_t n, uint64_t key1, uint64_t key2,
                                  uint64_t counter1, uint64_t counter2,
                                  double* data )
{
    TOAST_AUTO_TIMER();
    toast::rng::dist_uniform_01(n, key1, key2, counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_11 ( size_t n, uint64_t key1, uint64_t key2,
                                  uint64_t counter1, uint64_t counter2,
                                  double* data )
{
    TOAST_AUTO_TIMER();
    toast::rng::dist_uniform_11(n, key1, key2, counter1, counter2, data);
    return;
}

void ctoast_rng_dist_normal ( size_t n, uint64_t key1, uint64_t key2,
                              uint64_t counter1, uint64_t counter2,
                              double* data )
{
    TOAST_AUTO_TIMER();
    toast::rng::dist_normal(n, key1, key2, counter1, counter2, data);
    return;
}

//----------------------------------------------------------------------------//
// RNG (MT)

void ctoast_rng_dist_uint64_mt ( size_t blocks,      size_t n,
                                 uint64_t* key1,     uint64_t* key2,
                                 uint64_t* counter1, uint64_t* counter2,
                                 uint64_t* data ) {
    TOAST_AUTO_TIMER();
    toast::rng::mt::dist_uint64(blocks, n, key1, key2,
                                counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_01_mt ( size_t blocks,      size_t n,
                                     uint64_t* key1,     uint64_t* key2,
                                     uint64_t* counter1, uint64_t* counter2,
                                     double* data ) {
    TOAST_AUTO_TIMER();
    toast::rng::mt::dist_uniform_01(blocks, n, key1, key2,
                                    counter1, counter2, data );
    return;
}

void ctoast_rng_dist_uniform_11_mt ( size_t blocks,      size_t n,
                                     uint64_t* key1,     uint64_t* key2,
                                     uint64_t* counter1, uint64_t* counter2,
                                     double* data ) {
    TOAST_AUTO_TIMER();
    toast::rng::mt::dist_uniform_11(blocks, n, key1, key2,
                                    counter1, counter2, data );
    return;
}

void ctoast_rng_dist_normal_mt ( size_t blocks,      size_t n,
                                 uint64_t* key1,     uint64_t* key2,
                                 uint64_t* counter1, uint64_t* counter2,
                                 double* data ) {
    TOAST_AUTO_TIMER();
    toast::rng::mt::dist_normal(blocks, n, key1, key2,
                                counter1, counter2, data );
    return;
}


// Quaternion array

void ctoast_qarray_list_dot ( size_t n, size_t m, size_t d, double const * a,
                              double const * b, double * dotprod ) {
    TOAST_AUTO_TIMER();
    toast::qarray::list_dot(n, m, d, a, b, dotprod );
    return;
}

void ctoast_qarray_inv ( size_t n, double * q ) {
    TOAST_AUTO_TIMER();
    toast::qarray::inv( n, q );
    return;
}

void ctoast_qarray_amplitude ( size_t n, size_t m, size_t d,
                               double const * v, double * norm ) {
    TOAST_AUTO_TIMER();
    toast::qarray::amplitude( n, m, d, v, norm );
    return;
}

void ctoast_qarray_normalize ( size_t n, size_t m, size_t d,
                               double const * q_in, double * q_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::normalize( n, m, d, q_in, q_out );
    return;
}

void ctoast_qarray_normalize_inplace ( size_t n, size_t m, size_t d,
                                       double * q ) {
    TOAST_AUTO_TIMER();
    toast::qarray::normalize_inplace( n, m, d, q );
    return;
}

void ctoast_qarray_rotate ( size_t nq, double const * q, size_t nv,
                            double const * v_in, double * v_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::rotate(nq, q, nv, v_in, v_out );
    return;
}

void ctoast_qarray_mult ( size_t np, double const * p, size_t nq,
                          double const * q, double * r ) {
    TOAST_AUTO_TIMER();
    toast::qarray::mult(np, p, nq, q, r );
    return;
}

void ctoast_qarray_slerp ( size_t n_time, size_t n_targettime,
                           double const * time, double const * targettime,
                           double const * q_in, double * q_interp ) {
    TOAST_AUTO_TIMER();
    toast::qarray::slerp(n_time, n_targettime, time,
                         targettime, q_in, q_interp );
    return;
}

void ctoast_qarray_exp ( size_t n, double const * q_in, double * q_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::exp(n, q_in, q_out );
    return;
}

void ctoast_qarray_ln ( size_t n, double const * q_in, double * q_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::ln(n, q_in, q_out );
    return;
}

void ctoast_qarray_pow ( size_t n, double const * p, double const * q_in,
                         double * q_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::pow(n, p, q_in, q_out );
    return;
}

void ctoast_qarray_from_axisangle ( size_t n, double const * axis,
                                    double const * angle, double * q_out ) {
    TOAST_AUTO_TIMER();
    toast::qarray::from_axisangle(n, axis, angle, q_out );
    return;
}

void ctoast_qarray_to_axisangle ( size_t n, double const * q,
                                  double * axis, double * angle ) {
    TOAST_AUTO_TIMER();
    toast::qarray::to_axisangle(n, q, axis, angle );
    return;
}

void ctoast_qarray_to_rotmat ( double const * q, double * rotmat ) {
    TOAST_AUTO_TIMER();
     toast::qarray::to_rotmat( q, rotmat );
    return;
}

void ctoast_qarray_from_rotmat ( const double * rotmat, double * q ) {
    TOAST_AUTO_TIMER();
    toast::qarray::from_rotmat( rotmat, q );
    return;
}

void ctoast_qarray_from_vectors ( size_t n, double const * vec1,
    double const * vec2, double * q ) {
    TOAST_AUTO_TIMER();
    toast::qarray::from_vectors(n, vec1, vec2, q );
    return;
}

void ctoast_qarray_from_angles ( size_t n, double const * theta,
                                 double const * phi,
                                 double * const pa, double * quat, int IAU ) {
    TOAST_AUTO_TIMER();
    toast::qarray::from_angles(n, theta, phi, pa, quat, (IAU != 0) );
    return;
}

void ctoast_qarray_to_angles ( size_t n, double const * quat, double * theta,
                               double * phi, double * pa, int IAU ) {
    TOAST_AUTO_TIMER();
    toast::qarray::to_angles(n, quat, theta, phi, pa, (IAU != 0) );
    return;
}

void ctoast_qarray_to_position ( size_t n, double const * quat, double * theta,
                                 double * phi ) {
    TOAST_AUTO_TIMER();
    toast::qarray::to_position(n, quat, theta, phi );
    return;
}

// FFT

toast::fft::plan_type ctoast::convert_from_c ( ctoast_fft_plan_type in ) {
    toast::fft::plan_type ret;
    switch ( in ) {
        case PLAN_FAST :
            ret = toast::fft::plan_type::fast;
            break;
        case PLAN_BEST :
            ret = toast::fft::plan_type::best;
            break;
        default :
            TOAST_THROW( "invalid ctoast_fft_plan_type value" );
            break;
    }
    return ret;
}

ctoast_fft_plan_type ctoast::convert_to_c ( toast::fft::plan_type in ) {

    ctoast_fft_plan_type ret;
    switch ( in ) {
        case toast::fft::plan_type::fast :
            ret = PLAN_FAST;
            break;
        case toast::fft::plan_type::best :
            ret = PLAN_BEST;
            break;
        default :
            TOAST_THROW( "invalid toast::fft::plan_type value" );
            break;
    }
    return ret;
}

toast::fft::direction ctoast::convert_from_c ( ctoast_fft_direction in ) {
    toast::fft::direction ret;
    switch ( in ) {
        case FORWARD :
            ret = toast::fft::direction::forward;
            break;
        case BACKWARD :
            ret = toast::fft::direction::backward;
            break;
        default :
            TOAST_THROW( "invalid ctoast_fft_direction value" );
            break;
    }
    return ret;
}

ctoast_fft_direction ctoast::convert_to_c ( toast::fft::direction in ) {
    ctoast_fft_direction ret;
    switch ( in ) {
        case toast::fft::direction::forward :
            ret = FORWARD;
            break;
        case toast::fft::direction::backward :
            ret = BACKWARD;
            break;
        default :
            TOAST_THROW( "invalid toast::fft::direction value" );
            break;
    }
    return ret;
}

ctoast_fft_r1d * ctoast_fft_r1d_alloc ( int64_t length, int64_t n,
                                        ctoast_fft_plan_type type,
                                        ctoast_fft_direction dir, double scale )
{
    TOAST_AUTO_TIMER();
    toast::fft::direction tdir = ctoast::convert_from_c ( dir );
    toast::fft::plan_type ttype = ctoast::convert_from_c ( type );
    return reinterpret_cast < ctoast_fft_r1d * > ( toast::fft::r1d::create ( length, n, ttype, tdir, scale ) );
}

void ctoast_fft_r1d_free ( ctoast_fft_r1d * frd ) {
    delete reinterpret_cast < toast::fft::r1d * > ( frd );
    return;
}

int64_t ctoast_fft_r1d_length ( ctoast_fft_r1d * frd ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    return rd->length();
}

int64_t ctoast_fft_r1d_count ( ctoast_fft_r1d * frd ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    return rd->count();
}

double ** ctoast_fft_r1d_tdata ( ctoast_fft_r1d * frd ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > tdata = rd->tdata();
    double ** ret = (double**) malloc ( tdata.size() * sizeof(double*) );
    if ( ret == NULL ) {
        TOAST_THROW( "Cannot allocate fft_r1d_tdata buffer" );
    }
    for ( size_t i = 0; i < tdata.size(); ++i ) {
        ret[i] = tdata[i];
    }
    return ret;
}

void ctoast_fft_r1d_tdata_set ( ctoast_fft_r1d * frd, double ** data ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    int64_t len = rd->length();
    std::vector < double * > tdata = rd->tdata();
    for ( size_t i = 0; i < tdata.size(); ++i ) {
        std::copy ( data[i], data[i] + len, tdata[i] );
    }
    return;
}

void ctoast_fft_r1d_tdata_get ( ctoast_fft_r1d * frd, double ** data ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > tdata = rd->tdata();
    int64_t len = rd->length();
    for ( size_t i = 0; i < tdata.size(); ++i ) {
        std::copy ( tdata[i], tdata[i] + len, data[i] );
    }
    return;
}

double ** ctoast_fft_r1d_fdata ( ctoast_fft_r1d * frd ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > fdata = rd->fdata();
    double ** ret = (double**) malloc ( fdata.size() * sizeof(double*) );
    if ( ret == NULL ) {
        TOAST_THROW( "Cannot allocate fft_r1d_fdata buffer" );
    }
    for ( size_t i = 0; i < fdata.size(); ++i ) {
        ret[i] = fdata[i];
    }
    return ret;
}

void ctoast_fft_r1d_fdata_set ( ctoast_fft_r1d * frd, double ** data ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    int64_t len = rd->length();
    std::vector < double * > fdata = rd->fdata();
    for ( size_t i = 0; i < fdata.size(); ++i ) {
        std::copy ( data[i], data[i] + len, fdata[i] );
    }
    return;
}

void ctoast_fft_r1d_fdata_get ( ctoast_fft_r1d * frd, double ** data ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    std::vector < double * > fdata = rd->fdata();
    int64_t len = rd->length();
    for ( size_t i = 0; i < fdata.size(); ++i ) {
        std::copy ( fdata[i], fdata[i] + len, data[i] );
    }
    return;
}

void ctoast_fft_r1d_exec ( ctoast_fft_r1d * frd ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d * rd = reinterpret_cast < toast::fft::r1d * > ( frd );
    rd->exec();
    return;
}

ctoast_fft_r1d_plan_store * ctoast_fft_r1d_plan_store_get ( ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d_plan_store & st = toast::fft::r1d_plan_store::get();
    return reinterpret_cast < ctoast_fft_r1d_plan_store * > ( &st );
}

void ctoast_fft_r1d_plan_store_clear ( ctoast_fft_r1d_plan_store * pstore ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    st->clear();
    return;
}

void ctoast_fft_r1d_plan_store_cache ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    st->cache ( len, n );
    return;
}

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_forward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    toast::fft::r1d_p rdp = st->forward ( len, n );
    toast::fft::r1d * rd = rdp.get();
    return reinterpret_cast < ctoast_fft_r1d * > ( rd );
}

ctoast_fft_r1d * ctoast_fft_r1d_plan_store_backward ( ctoast_fft_r1d_plan_store * pstore, int64_t len, int64_t n ) {
    TOAST_AUTO_TIMER();
    toast::fft::r1d_plan_store * st = reinterpret_cast < toast::fft::r1d_plan_store * > ( pstore );
    toast::fft::r1d_p rdp = st->backward ( len, n );
    toast::fft::r1d * rd = rdp.get();
    return reinterpret_cast < ctoast_fft_r1d * > ( rd );
}

// Healpix

void ctoast_healpix_ang2vec ( int64_t n, double const * theta, double const * phi, double * vec ) {
    TOAST_AUTO_TIMER();
    toast::healpix::ang2vec ( n, theta, phi, vec );
    return;
}

void ctoast_healpix_vec2ang ( int64_t n, double const * vec, double * theta, double * phi ) {
    TOAST_AUTO_TIMER();
    toast::healpix::vec2ang ( n, vec, theta, phi );
    return;
}

void ctoast_healpix_vecs2angpa ( int64_t n, double const * vec, double * theta, double * phi, double * pa ) {
    TOAST_AUTO_TIMER();
    toast::healpix::vecs2angpa ( n, vec, theta, phi, pa );
    return;
}

ctoast_healpix_pixels * ctoast_healpix_pixels_alloc ( int64_t nside ) {
    return reinterpret_cast < ctoast_healpix_pixels * > ( new toast::healpix::pixels ( nside ) );
}

void ctoast_healpix_pixels_free ( ctoast_healpix_pixels * hpix ) {
    delete reinterpret_cast < toast::healpix::pixels * > ( hpix );
    return;
}

void ctoast_healpix_pixels_reset ( ctoast_healpix_pixels * hpix, int64_t nside ) {
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->reset ( nside );
    return;
}

void ctoast_healpix_pixels_vec2zphi ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, double * phi, int * region, double * z, double * rtz ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2zphi ( n, vec, phi, region, z, rtz );
    return;
}

void ctoast_healpix_pixels_theta2z ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, int * region, double * z, double * rtz ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->theta2z ( n, theta, region, z, rtz );
    return;
}

void ctoast_healpix_pixels_zphi2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * phi, int const * region, double const * z, double const * rtz, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->zphi2nest ( n, phi, region, z, rtz, pix );
    return;
}

void ctoast_healpix_pixels_zphi2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * phi, int const * region, double const * z, double const * rtz, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->zphi2ring ( n, phi, region, z, rtz, pix );
    return;
}

void ctoast_healpix_pixels_ang2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, double const * phi, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ang2nest ( n, theta, phi, pix );
    return;
}

void ctoast_healpix_pixels_ang2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * theta, double const * phi, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ang2ring ( n, theta, phi, pix );
    return;
}

void ctoast_healpix_pixels_vec2nest ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2nest ( n, vec, pix );
    return;
}

void ctoast_healpix_pixels_vec2ring ( ctoast_healpix_pixels * hpix, int64_t n, double const * vec, int64_t * pix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->vec2ring ( n, vec, pix );
    return;
}

void ctoast_healpix_pixels_ring2nest ( ctoast_healpix_pixels * hpix, int64_t n, int64_t const * ringpix, int64_t * nestpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->ring2nest ( n, ringpix, nestpix );
    return;
}

void ctoast_healpix_pixels_nest2ring ( ctoast_healpix_pixels * hpix, int64_t n, int64_t const * nestpix, int64_t * ringpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->nest2ring ( n, nestpix, ringpix );
    return;
}

void ctoast_healpix_pixels_degrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->degrade_ring ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_degrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->degrade_nest ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_upgrade_ring ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->upgrade_ring ( factor, n, inpix, outpix );
    return;
}

void ctoast_healpix_pixels_upgrade_nest ( ctoast_healpix_pixels * hpix, int factor, int64_t n, int64_t const * inpix, int64_t * outpix ) {
    TOAST_AUTO_TIMER();
    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );
    hp->upgrade_nest ( factor, n, inpix, outpix );
    return;
}

//--------------------------------------
// Operator helpers
//--------------------------------------

char ** ctoast_string_alloc ( size_t nstring, size_t length ) {
    char ** ret = (char**) malloc( nstring * sizeof(char*) );

    if ( ! ret ) {
        std::ostringstream o;
        o << "failed to allocate array of " << nstring << " C strings";
        TOAST_THROW( o.str().c_str() );
    }

    for ( size_t i = 0; i < nstring; ++i ) {
        ret[i] = (char*) malloc ( (length + 1) * sizeof(char) );
        if ( ! ret[i] ) {
            std::ostringstream o;
            o << "failed to allocate C string of " << (length+1) << " characters";
            TOAST_THROW( o.str().c_str() );
        }
    }

    return ret;
}

void ctoast_string_free ( size_t nstring, char ** str ) {
    if ( str != NULL ) {
        for ( size_t i = 0; i < nstring; ++i ) {
            if ( str[i] != NULL ) {
                free( str[i] );
            }
        }
        free(str);
    }
    return;
}

//--------------------------------------
// Atmosphere sub-library
//--------------------------------------

ctoast_atm_sim * ctoast_atm_sim_alloc (
    double azmin, double azmax, double elmin, double elmax, double tmin,
    double tmax, double lmin_center, double lmin_sigma, double lmax_center,
    double lmax_sigma, double w_center, double w_sigma, double wdir_center,
    double wdir_sigma, double z0_center, double z0_sigma, double T0_center,
    double T0_sigma, double zatm, double zmax, double xstep, double ystep,
    double zstep, long nelem_sim_max, int verbosity, MPI_Comm comm,
    int gangsize, uint64_t key1, uint64_t key2, uint64_t counter1,
    uint64_t counter2, char *cachedir ) {

#ifdef HAVE_ELEMENTAL
    TOAST_AUTO_TIMER();
    try {
        return reinterpret_cast < ctoast_atm_sim * >
            (new toast::tatm::sim ( azmin, azmax, elmin, elmax, tmin, tmax,
                                    lmin_center, lmin_sigma, lmax_center,
                                    lmax_sigma, w_center, w_sigma, wdir_center,
                                    wdir_sigma, z0_center, z0_sigma, T0_center,
                                    T0_sigma, zatm, zmax, xstep, ystep, zstep,
                                    nelem_sim_max, verbosity, comm, gangsize,
                                    key1, key2, counter1, counter2, cachedir ) );
    } catch ( std::exception &e ){
        std::cerr << "ERROR allocating atmosphere: " << e.what() << std::endl;
    } catch ( ... ) {
        std::cerr << "unknown ERROR allocating atmosphere. " << std::endl;
    }
#endif
    return NULL;
}

int ctoast_atm_sim_free ( ctoast_atm_sim * sim ) {
#ifdef HAVE_ELEMENTAL
    try {
            delete reinterpret_cast < toast::tatm::sim * > ( sim );
    } catch ( std::exception &e ){
        std::cerr << "ERROR freeing atmosphere: " << e.what() << std::endl;
        return -1;
    } catch ( ... ) {
        std::cerr << "unknown ERROR freeing atmosphere. " << std::endl;
        return -1;
    }
#endif
    return 0;
}

int ctoast_atm_sim_simulate( ctoast_atm_sim * sim, int use_cache ) {
#ifdef HAVE_ELEMENTAL
    TOAST_AUTO_TIMER();
    toast::tatm::sim * sm = reinterpret_cast < toast::tatm::sim * > ( sim );
    int err;
    try {
        err = sm->simulate( (use_cache != 0) );
    } catch ( std::exception &e ) {
        std::cerr << "ERROR simulating the atmosphere: " << e.what()
                  << std::endl;
        return -1;
    } catch ( ... ) {
        std::cerr << "unknown ERROR simulating the atmosphere" << std::endl;
        return -1;
    }
    return err;
#else
    return -1;
#endif
}

int ctoast_atm_sim_observe(
    ctoast_atm_sim * sim, double *t, double *az, double *el,
    double *tod, long nsamp, double fixed_r ) {
#ifdef HAVE_ELEMENTAL
    TOAST_AUTO_TIMER();
    toast::tatm::sim * sm = reinterpret_cast < toast::tatm::sim * > ( sim );
    int err;
    try {
        err = sm->observe( t, az, el, tod, nsamp, fixed_r );
    } catch ( std::exception &e ) {
        std::cerr << "ERROR observing the atmosphere: " << e.what() << std::endl;
        return -1;
    } catch ( ... ) {
        std::cerr << "unknown ERROR observing the atmosphere" << std::endl;
        return -1;
    }
    return err;
#else
    return -1;
#endif
}

double ctoast_atm_get_absorption_coefficient(double altitude,
                                             double temperature,
                                             double pressure, double pwv,
                                             double freq) {
    double absorption = 0;
#ifdef HAVE_AATM
    TOAST_AUTO_TIMER();
    absorption = toast::tatm::get_absorption_coefficient(altitude, temperature,
                                                         pressure, pwv, freq);
#endif
    return absorption;
}

int ctoast_atm_get_absorption_coefficient_vec(double altitude,
                                              double temperature,
                                              double pressure, double pwv,
                                              double freqmin, double freqmax,
                                              size_t nfreq,
                                              double *absorption) {
    int err = 1;
#ifdef HAVE_AATM
    TOAST_AUTO_TIMER();
    err = toast::tatm::get_absorption_coefficient_vec(altitude, temperature,
                                                      pressure, pwv,
                                                      freqmin, freqmax, nfreq,
                                                      absorption);
#endif
    return err;
}

double ctoast_atm_get_atmospheric_loading(double altitude, double temperature,
                                          double pressure, double pwv,
                                          double freq) {
    double loading = 0;
#ifdef HAVE_AATM
    TOAST_AUTO_TIMER();
    loading = toast::tatm::get_atmospheric_loading(altitude, temperature,
                                                   pressure, pwv, freq);

#endif
    return loading;
}

int ctoast_atm_get_atmospheric_loading_vec(double altitude, double temperature,
                                           double pressure, double pwv,
                                           double freqmin, double freqmax,
                                           size_t nfreq, double *loading) {
    int err = 1;
#ifdef HAVE_AATM
    TOAST_AUTO_TIMER();
    err = toast::tatm::get_atmospheric_loading_vec(altitude, temperature,
                                                   pressure, pwv,
                                                   freqmin, freqmax, nfreq,
                                                   loading);
#endif
    return err;
}

//--------------------------------------
// TOD sub-library
//--------------------------------------

void ctoast_pointing_healpix_matrix ( ctoast_healpix_pixels * hpix, int nest,
    double eps, double cal, char const * mode, size_t n, double const * pdata,
    double const * hwpang, uint8_t const * flags, int64_t * pixels,
    double * weights ) {
    TOAST_AUTO_TIMER();

    toast::healpix::pixels * hp = reinterpret_cast < toast::healpix::pixels * > ( hpix );

    std::string modestr(mode);
    bool bnest = true;
    if ( nest == 0 ) {
        bnest = false;
    }

    toast::pointing::healpix_matrix ( (*hp), bnest, eps, cal, modestr, n,
        pdata, hwpang, flags, pixels, weights );

    return;
}

void ctoast_filter_polyfilter (
    const long order, double **signals, uint8_t *flags,
    const size_t n, const size_t nsignal,
    const long *starts, const long *stops, const size_t nscan ) {
    TOAST_AUTO_TIMER();

    toast::filter::polyfilter ( order, signals, flags, n, nsignal,
                                starts, stops, nscan );

    return;
}

void ctoast_sim_noise_sim_noise_timestream (
    const uint64_t realization, const uint64_t telescope,
    const uint64_t component, const uint64_t obsindx, const uint64_t detindx,
    const double rate, const uint64_t firstsamp, const uint64_t samples,
    const uint64_t oversample, const double *freq, const double *psd,
    const uint64_t psdlen, double *noise ) {

    TOAST_AUTO_TIMER();
    toast::sim_noise::sim_noise_timestream(
        realization, telescope, component, obsindx, detindx, rate, firstsamp,
        samples, oversample, freq, psd, psdlen, noise);
}



void ctoast_sim_map_scan_map32 (
    long *submap, long subnpix, double *weights, size_t nmap, long *subpix,
    float *map, double *tod, size_t nsamp ) {

    TOAST_AUTO_TIMER();
    toast::sim_map::scan_map ( submap, subnpix, weights, nmap, subpix, map,
                               tod, nsamp );

    return;
}

void ctoast_sim_map_scan_map64 (
    long *submap, long subnpix, double *weights, size_t nmap, long *subpix,
    double *map, double *tod, size_t nsamp ) {

    TOAST_AUTO_TIMER();
    toast::sim_map::scan_map ( submap, subnpix, weights, nmap, subpix, map,
                               tod, nsamp );

    return;
}

//--------------------------------------
// FOD sub-library
//--------------------------------------

void ctoast_fod_autosums ( int64_t n, double const * x, uint8_t const * good, int64_t lagmax, double * sums, int64_t * hits ) {
    TOAST_AUTO_TIMER();
    toast::fod::autosums ( n, x, good, lagmax, sums, hits );
    return;
}

void ctoast_fod_crosssums ( int64_t n, double const * x, double const * y, uint8_t const * good, int64_t lagmax, double * sums, int64_t * hits ) {
    TOAST_AUTO_TIMER();
    toast::fod::crosssums ( n, x, y, good, lagmax, sums, hits );
    return;
}


//--------------------------------------
// Map sub-library
//--------------------------------------

void ctoast_cov_accumulate_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, double const * signal, double * zdata, int64_t * hits, double * invnpp ) {
    TOAST_AUTO_TIMER();
    toast::cov::accumulate_diagonal ( nsub, subsize, nnz, nsamp, indx_submap, indx_pix,
        weights, scale, signal, zdata, hits, invnpp );
    return;
}

void ctoast_cov_accumulate_diagonal_hits ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, int64_t * hits ) {
    TOAST_AUTO_TIMER();
    toast::cov::accumulate_diagonal_hits ( nsub, subsize, nnz, nsamp, indx_submap,
        indx_pix, hits );
    return;
}

void ctoast_cov_accumulate_diagonal_invnpp ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, int64_t * hits, double * invnpp ) {
    TOAST_AUTO_TIMER();
    toast::cov::accumulate_diagonal_invnpp ( nsub, subsize, nnz, nsamp, indx_submap, indx_pix,
        weights, scale, hits, invnpp );
    return;
}

void ctoast_cov_accumulate_zmap ( int64_t nsub, int64_t subsize, int64_t nnz, int64_t nsamp,
    int64_t const * indx_submap, int64_t const * indx_pix, double const * weights,
    double scale, double const * signal, double * zdata ) {
    TOAST_AUTO_TIMER();
    toast::cov::accumulate_zmap ( nsub, subsize, nnz, nsamp, indx_submap, indx_pix,
        weights, scale, signal, zdata );
    return;
}

void ctoast_cov_eigendecompose_diagonal ( int64_t nsub, int64_t subsize,
    int64_t nnz, double * data, double * cond, double threshold,
    int32_t do_invert, int32_t do_rcond ) {
    TOAST_AUTO_TIMER();
    toast::cov::eigendecompose_diagonal ( nsub, subsize, nnz, data, cond,
        threshold, do_invert, do_rcond );
    return;
}

void ctoast_cov_multiply_diagonal ( int64_t nsub, int64_t subsize,
    int64_t nnz, double * data1, double const * data2 ) {
    TOAST_AUTO_TIMER();
    toast::cov::multiply_diagonal ( nsub, subsize, nnz, data1, data2 );
    return;
}

void ctoast_cov_apply_diagonal ( int64_t nsub, int64_t subsize, int64_t nnz,
    double const * mat, double * vec ) {
    TOAST_AUTO_TIMER();
    toast::cov::apply_diagonal ( nsub, subsize, nnz, mat, vec );
    return;
}


//--------------------------------------
// Run test suite
//--------------------------------------

int ctoast_test_runner ( int argc, char *argv[] ) {
    return toast::test::runner ( argc, argv );
}
