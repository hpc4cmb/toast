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

int32_t toast::util::timing_manager::fgMaxDepth =
        std::numeric_limits<uint16_t>::max();

//============================================================================//
// static function
toast::util::timing_manager* toast::util::timing_manager::instance()
{
    if(!fgInstance) new toast::util::timing_manager();
    return fgInstance;
}

//============================================================================//

#if defined(DISABLE_TIMERS)
bool toast::util::timing_manager::fgEnabled = false;
#else
bool toast::util::timing_manager::fgEnabled = true;
#endif

//============================================================================//
// static function
void toast::util::timing_manager::enable(bool val)
{
#if defined(DISABLE_TIMERS)
    val = false;
#endif
    fgEnabled = val;
}

//============================================================================//
// static function
toast::util::timing_manager::comm_group_t
toast::util::timing_manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    const int32_t nthreads = toast::get_env<int32_t>("OMP_NUM_THREADS", 1);
    int32_t max_processes = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi_size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count = toast::get_env<int32_t>("TOAST_NODE_COUNT",
                                                     mpi_node_default);
    int32_t mpi_split_size = mpi_rank() / (mpi_size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    MPI_Comm local_mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_split_size, mpi_rank(), &local_mpi_comm);

#if defined(DEBUG)
    int32_t local_mpi_rank = mpi_rank(local_mpi_comm);
    int32_t local_mpi_size = mpi_size(local_mpi_comm);
    int32_t local_mpi_file = mpi_rank() / local_mpi_size;

    printf("WORLD RANK/SIZE: %d/%d --> ROW RANK/SIZE: %d/%d\n",
        mpi_rank(), mpi_size(), local_mpi_rank, local_mpi_size);

    std::stringstream _info;
    _info << mpi_rank() << " Rank      : " << mpi_rank() << std::endl;
    _info << mpi_rank() << " Node      : " << mpi_node_count << std::endl;
    _info << mpi_rank() << " Local Size: " << local_mpi_size << std::endl;
    _info << mpi_rank() << " Local Rank: " << local_mpi_rank << std::endl;
    _info << mpi_rank() << " Local File: " << local_mpi_file << std::endl;
    std::cout << _info.str();
#endif

    return comm_group_t(local_mpi_comm, mpi_rank() / mpi_size(local_mpi_comm));
}

//============================================================================//
// static function
void toast::util::timing_manager::write_json(string_t _fname)
{
    const int32_t mpi_root = 0;
    comm_group_t mpi_comm_group = get_communicator_group();
    MPI_Comm& local_mpi_comm = std::get<0>(mpi_comm_group);
    int32_t local_mpi_file = std::get<1>(mpi_comm_group);

    {
        std::stringstream _rss;
        _rss << "_" << local_mpi_file;
        _fname.insert(_fname.find_last_of("."), _rss.str());
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "[" << mpi_rank() << "] Writing serialization file: "
              << _fname << std::endl;
        std::cout << _info.str();
    }

    // output stream
    std::stringstream fss;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::tab;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 1);
        cereal::JSONOutputArchive oa(fss, opts);

        oa(cereal::make_nvp("timing_manager", *timing_manager::instance()));
    }

    // if another entry follows
    if(mpi_rank(local_mpi_comm)+1 < mpi_size(local_mpi_comm))
        fss << ",";

    // the JSON output as a string
    string_t fss_str = fss.str();
    // limit the iteration loop. Occasionally it seems that this will create
    // an infinite loop even though it shouldn't...
    const uint64_t itr_limit = fss_str.length();
    // compact the JSON
    for(auto citr : { "\n", "\t", "  " })
    {
        string_t itr(citr);
        string_t::size_type fpos = 0;
        uint64_t nitr = 0;
        do
        {
            fpos = fss_str.find(itr, fpos);
            if(fpos != string_t::npos)
                fss_str.replace(fpos, itr.length(), " ");
            ++nitr;
        }
        while(nitr < itr_limit && fpos != string_t::npos);
    }

    // now we need to gather the lengths of each serialization string
    int fss_len = fss_str.length();
    int* recvcounts = nullptr;

    // Only root has the received data
    if (mpi_rank(local_mpi_comm) == mpi_root)
        recvcounts = (int*) malloc( mpi_size(local_mpi_comm) * sizeof(int)) ;

    MPI_Gather(&fss_len, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               mpi_root, local_mpi_comm);

    // Figure out the total length of string, and displacements for each rank
    int fss_tot_len = 0;
    int* fss_tot = nullptr;
    char* totalstring = nullptr;

    if (mpi_rank(local_mpi_comm) == mpi_root)
    {
        fss_tot = (int*) malloc( mpi_size(local_mpi_comm) * sizeof(int) );

        fss_tot[0] = 0;
        fss_tot_len += recvcounts[0]+1;

        for(int32_t i = 1; i < mpi_size(local_mpi_comm); ++i)
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

    char* cfss = (char*) fss_str.c_str();
    MPI_Gatherv(cfss, fss_len, MPI_CHAR,
                totalstring, recvcounts, fss_tot, MPI_CHAR,
                mpi_root, local_mpi_comm);

    if (mpi_rank(local_mpi_comm) == mpi_root)
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

    MPI_Comm_free(&local_mpi_comm);
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

toast::util::timing_manager::string_t
toast::util::timing_manager::get_prefix() const
{
    static string_t* _prefix = nullptr;
    if(!_prefix)
    {
        // prefix spacing
        static uint16_t width = 1;
        if(mpi_size() > 9)
            width = std::max(width, (uint16_t) ( log10(mpi_size()) + 1 ));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << mpi_rank() << "> ";
        _prefix = new string_t(ss.str());
    }
    return *_prefix;
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

    uint64_t ref = (string_hash(key) + string_hash(tag)) * (ncount+1) * (nhash+1);

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
    // designated as [cxx], [pyc], etc.
    ss << get_prefix() << "[" << tag << "] ";

    // indent
    for(int64_t i = 0; i < ncount; ++i)
        ss << "| ";

    ss << std::left << key;
    toast::util::timer::propose_output_width(ss.str().length());

    m_timer_map[ref] = toast_timer_t(3, ss.str(), string_t(""));

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    timer_tuple_t _tuple(ref, ncount, tag_ss.str(), m_timer_map[ref]);
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
        if(_os == &std::cout || _os == &std::cerr)
            return;
        ofstream_t* fos = get_ofstream(_os);
        if(!(fos->is_open() && fos->good()))
        {
            std::cerr << "Output stream for " << id << " is not open/valid. "
                      << "Redirecting to stdout..." << std::endl;
            _os = &std::cout;
        }
    };

    if(os == m_report)
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
            delete (ofstream_t*) m_os;

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
