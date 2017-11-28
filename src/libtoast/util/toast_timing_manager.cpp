/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include "toast_util_internal.hpp"
#include "toast_math_internal.hpp"

#include <sstream>
#include <algorithm>

//============================================================================//

CEREAL_CLASS_VERSION(toast::util::timer_tuple, TOAST_TIMER_VERSION)
CEREAL_CLASS_VERSION(toast::util::timing_manager, TOAST_TIMER_VERSION)

//============================================================================//

toast::util::timing_manager* toast::util::timing_manager::fgInstance = nullptr;

//============================================================================//
// static function
toast::util::timing_manager* toast::util::timing_manager::instance()
{
    if(!fgInstance) new toast::util::timing_manager();
	return fgInstance;
}

//============================================================================//
// static function
void toast::util::timing_manager::write_json(const string_t& _fname)
{
    // output stream
    std::stringstream fss;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto _space = cereal::JSONOutputArchive::Options::IndentChar::space;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, _space, 2);
        cereal::JSONOutputArchive oa(fss, opts);

        oa(cereal::make_nvp("timing_manager", *timing_manager::instance()));
    }

    // if another entry follows
    if(mpi_rank()+1 < mpi_size())
        fss << ",";

    int fss_len = fss.str().length();
    const int mpi_root = 0;
    int* recvcounts = nullptr;

    // Only root has the received data
    if (mpi_rank() == mpi_root)
        recvcounts = (int*) malloc( mpi_size() * sizeof(int)) ;

    MPI_Gather(&fss_len, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               mpi_root, MPI_COMM_WORLD);

    // Figure out the total length of string, and displacements for each rank
    int fss_tot_len = 0;
    int* fss_tot = nullptr;
    char* totalstring = nullptr;

    if (mpi_rank() == mpi_root)
    {
        fss_tot = (int*) malloc( mpi_size() * sizeof(int) );

        fss_tot[0] = 0;
        fss_tot_len += recvcounts[0]+1;

        for(int32_t i = 1; i < mpi_size(); ++i)
        {
            // plus one for space or \0 after words
            fss_tot_len += recvcounts[i]+1;
            fss_tot[i] = fss_tot[i-1] + recvcounts[i-1] + 1;
        }

        // allocate string, pre-fill with spaces and null terminator
        totalstring = (char*) malloc(fss_tot_len * sizeof(char));
        for(int32_t i = 0; i < fss_tot_len-1; ++i)
            totalstring[i] = ' ';
        totalstring[fss_tot_len-1] = '\0';
    }

    // Now we have the receive buffer, counts, and displacements, and
    // can gather the strings

    char* cfss = (char*) fss.str().c_str();
    MPI_Gatherv(cfss, fss_len, MPI_CHAR,
                totalstring, recvcounts, fss_tot, MPI_CHAR,
                mpi_root, MPI_COMM_WORLD);


    if (mpi_rank() == mpi_root)
    {
        ofstream_t ofs;
        ofs.open(_fname);
        ofs << "{\n\"ranks\": [" << std::endl;
        ofs << totalstring << std::endl;
        ofs << "]" << "\n}" << std::endl;
        free(totalstring);
        free(fss_tot);
        free(recvcounts);
    }

}

//============================================================================//

toast::util::timing_manager::timing_manager()
: m_report(&std::cout)
{
	if(!fgInstance) { fgInstance = this; }
    else
    {
        std::ostringstream ss;
        ss << "toast::util::timing_manager singleton has already been created";
        TOAST_THROW( ss.str().c_str() );
    }
}

//============================================================================//

toast::util::timing_manager::~timing_manager()
{
    auto close_ostream = [&] (ostream_t*& m_os)
    {
        ofstream_t* m_fos = get_ofstream(m_os);
        if(!m_fos->good() || !m_fos->is_open())
            return;
        m_fos->close();
    };

    close_ostream(m_report);

    fgInstance = nullptr;
}

//============================================================================//

toast::util::timer& toast::util::timing_manager::timer(const string_t& key,
                                                       const string_t& tag,
                                                       int32_t ncount,
                                                       int32_t nhash)
{
#if defined(DEBUG)
    if(key.find(" ") != string_t::npos)
    {
        std::stringstream ss;
        ss << "Error! Space found in tag: \"" << key << "\"";
        throw std::runtime_error(ss.str().c_str());
    }
#endif

    uint64_t sum = 0;
    for(size_t i = 0; i < key.length(); ++i)
        sum += (int64_t) key[i];
    for(size_t i = 0; i < tag.length(); ++i)
        sum += (int64_t) tag[i];

    uint64_t ref = (string_hash(key) + string_hash(tag)) * (ncount+1) * nhash;

    // if already exists, return it
    if(m_timer_map.find(ref) != m_timer_map.end())
        return m_timer_map.find(ref)->second;

    // special case of auto_timer as the first timer
    if(ncount == 1 && m_timer_list.size() == 0)
    {
        toast::util::details::base_timer::get_instance_count()--;
        ncount = 0;
    }

    std::stringstream ss;
    ss << "> " << "[" << tag << "] "; // designated as [cxx], [pyc], etc.

    // indent
    for(int64_t i = 0; i < ncount; ++i)
        ss << "  ";

    ss << std::left << key;
    toast::util::timer::propose_output_width(ss.str().length());

    m_timer_map[ref] = toast_timer_t(3, ss.str(), string_t(""));

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    timer_tuple_t _tuple(ref, tag_ss.str(), m_timer_map[ref]);
    m_timer_list.push_back(_tuple);

    return m_timer_map[ref];
}

//============================================================================//

void toast::util::timing_manager::report() const
{
    for(int32_t i = 0; i < mpi_size(); ++i)
    {
        // blocking
        if(mpi_is_initialized())
            MPI_Barrier(MPI_COMM_WORLD);
        // only 1 at a time
        if( i != mpi_rank() )
            continue;
        report(m_report);
    }
}

//============================================================================//

void toast::util::timing_manager::report(ostream_t* os) const
{
    auto check_stream = [&] (ostream_t*& _os, const string_t& id)
    {
        if(_os == &std::cout)
            return;
        ofstream_t* fos = get_ofstream(_os);
        if(!(fos->is_open() && fos->good()))
        {
            std::cerr << "Output stream for " << id << " is not open/valid. "
                      << "Redirecting to stdout..." << std::endl;
            _os = &std::cout;
        }
    };

    check_stream(os, "total timing report");

    for(const auto& itr : *this)
        itr.timer().stop();

    *os << "> rank " << mpi_rank() << std::endl;

    for(const auto& itr : *this)
        itr.timer().report(*os);

    os->flush();
}

//============================================================================//

void toast::util::timing_manager::set_output_stream(ostream_t& _os)
{
    m_report = &_os;
}

//============================================================================//

void toast::util::timing_manager::set_output_stream(const string_t& fname)
{
    auto ostreamop = [&] (ostream_t*& m_os, const string_t& _fname)
    {
        if(m_os != &std::cout)
            delete m_os;

        ofstream_t* _fos = new ofstream_t;
        for(int32_t i = 0; i < mpi_size(); ++i)
        {
            if(mpi_is_initialized())
                MPI_Barrier(MPI_COMM_WORLD);
            if(mpi_rank() != i)
                continue;

            if(mpi_rank() == 0)
                _fos->open(_fname);
            else
                _fos->open(_fname, std::ios_base::out | std::ios_base::app);
        }

        if(_fos->is_open() && _fos->good())
            m_os = _fos;
        else
        {
            std::cerr << "Warning! Unable to open file " << _fname << ". "
                      << "Redirecting to stdout..." << std::endl;
            _fos->close();
            delete _fos;
            m_os = &std::cout;
        }
    };

    ostreamop(m_report, fname);

}

//============================================================================//

toast::util::timing_manager::ofstream_t*
toast::util::timing_manager::get_ofstream(ostream_t* m_os) const
{
    return static_cast<ofstream_t*>(m_os);
}

//============================================================================//
