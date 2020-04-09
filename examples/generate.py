#!/usr/bin/env python3

import os
import sys
import stat
import re
import shutil

import argparse

import toml


def create_argparse_file(props, file):
    """
    Take a dictionary from the TOML and make an argparse compatible
    parameter file.
    """
    with open(file, "w") as f:
        for k, v in props.items():
            if isinstance(v, bool):
                f.write("--{}\n".format(k))
            else:
                f.write("--{}\n".format(k))
                f.write("{}\n".format(v))
    return


def create_telescope_commands(
    tele, focalplane, schedule, jobdir, mpi_launch, mpi_procs, mpi_nodes
):
    """
    Return lines of shell code to generate a focalplane and schedule for a telescope.
    """
    mpistr = "{}".format(mpi_launch)
    if mpi_procs != "":
        mpistr = "{} {} 1".format(mpistr, mpi_procs)
    if mpi_nodes != "":
        mpistr = "{} {} 1".format(mpistr, mpi_nodes)
    detpix = focalplane["minpix"]
    fp_par_file = "focalplane_{}_{}.par".format(tele, detpix)
    fp_file = "focalplane_{}_{}.pkl".format(tele, detpix)
    fp_root = "focalplane_{}".format(tele)

    create_argparse_file(focalplane, os.path.join(jobdir, fp_par_file))

    outstr = '# Create telescope model for "{}"\n'.format(tele)
    outstr = '{}if [ ! -e "{}" ]; then\n'.format(outstr, fp_file)
    outstr = '{}    echo "Creating focalplane {} ..."\n'.format(outstr, fp_file)
    outstr = '{}    {} toast_fake_focalplane.py @{} --out "{}"\n'.format(
        outstr, mpistr, fp_par_file, fp_root
    )
    outstr = "{}fi\n".format(outstr)

    if len(schedule) > 0:
        schd_par_file = "schedule_{}.par".format(tele)
        schd_file = "schedule_{}.txt".format(tele)
        create_argparse_file(schedule, os.path.join(jobdir, schd_par_file))
        outstr = '{}if [ ! -e "{}" ]; then\n'.format(outstr, schd_file)
        outstr = '{}    echo "Creating schedule {} ..."\n'.format(outstr, schd_file)
        outstr = '{}    {} toast_ground_schedule.py @{} --out "{}"\n'.format(
            outstr, mpistr, schd_par_file, schd_file
        )
        outstr = "{}fi\n".format(outstr)

    outstr = "{}\n".format(outstr)

    return outstr


def create_pysm_commands(
    mapfile,
    nside,
    bandcenter_ghz,
    bandwidth_ghz,
    beam_arcmin,
    coord,
    mpi_launch,
    mpi_procs,
    mpi_nodes,
):
    """
    Return lines of shell code to generate the precomputed input sky map.
    """
    mpistr = "{}".format(mpi_launch)
    if mpi_procs != "":
        mpistr = "{} {} 1".format(mpistr, mpi_procs)
    if mpi_nodes != "":
        mpistr = "{} {} 1".format(mpistr, mpi_nodes)

    outstr = "# Create sky model\n"
    outstr = '{}if [ ! -e "{}" ]; then\n'.format(outstr, mapfile)
    outstr = '{}    echo "Creating sky model {} ..."\n'.format(outstr, mapfile)
    outstr = '{}    {} ./pysm_sky.py --output "{}" --nside {} --bandcenter_ghz {} --bandwidth_ghz {} --beam_arcmin {} --coord {}\n'.format(
        outstr,
        mpistr,
        mapfile,
        nside,
        bandcenter_ghz,
        bandwidth_ghz,
        beam_arcmin,
        coord,
    )
    outstr = "{}fi\n".format(outstr)

    outstr = "{}\n".format(outstr)

    return outstr


def ceil_div(a, b):
    """Handy, from: https://stackoverflow.com/questions/14822184/
    """
    return -(-a // b)


def compute_job_size(
    max_nodes,
    node_mem,
    node_max_procs,
    omp_min_threads,
    obs_count,
    obs_mem,
    obs_max_procs,
    large_groups=False,
):
    """
    Given constraints of the system and the job, compute the process grouping.

    By default, this function tries to make the group size as small as possible based
    on memory constraints and then sets the number of groups to be exactly the number
    of observations (each group has one observation).

    If large_groups is True, then instead the starting point is to make the groups
    as large as possible given the maximum allowed for the observations.  The observations are then distributed among these large groups.

    Args:


    Returns:
        (tuple): the number of nodes, procs per node, threads per proc, and group size.

    """
    print(
        "      system has {} nodes, each with {} GB memory and up to {} procs".format(
            max_nodes, node_mem, node_max_procs
        )
    )
    print(
        "      {} observations, each requiring {} GB memory and using up to {} procs".format(
            obs_count, obs_mem, obs_max_procs
        )
    )
    nodes = None
    node_procs = node_max_procs
    omp_threads = omp_min_threads
    group_size = None
    total_procs = None

    if large_groups:
        print("      Using largest possible group size.")
        group_mem = None
        if obs_max_procs > node_procs:
            # Maximizing the observation processes requires multiple nodes
            print(
                "      Max observation procs > node procs, using one or more nodes per group"
            )
            group_nodes = obs_max_procs // node_procs
            group_size = group_nodes * node_procs
            group_mem = group_nodes * node_mem
        else:
            # See if we can fit multiple maximal observations on one node.
            print(
                "      Max observation procs <= node procs, using up to one node per group"
            )
            group_size = node_procs
            group_mem = node_mem
            while (group_size % 2 == 0) and (group_size >= (2 * obs_max_procs)):
                group_size = group_size // 2
                group_mem /= 2.0
        print(
            "      Largest group size = {} procs, {} procs per node".format(
                group_size, node_procs
            )
        )

        # If this largest group size exceeds the node count, reduce it until it fits
        while ceil_div(group_size, node_procs) > max_nodes:
            print(
                "      Group size of {} procs exceeds max node count ({}).  Adjusting to:".format(
                    group_size, max_nodes
                )
            )
            group_size -= node_procs
            group_mem -= node_mem
            print("        group size = {}".format(group_size))

        # Does this maximum group size have enough memory for even one observation?
        if group_mem < obs_mem:
            print(
                "      Required memory for one observation ({}) is greater than max ({}), skipping job".format(
                    obs_mem, group_mem
                )
            )
            return (None, None, None, None)

        # Compute the number of observations per group that will fit into memory
        print("      Group has {} GB of memory".format(group_mem))
        obs_per_group = int(group_mem / obs_mem)
        print("      Max observations per group = {}".format(obs_per_group))

        # Compute the number of groups needed to process all observations
        num_groups = ceil_div(obs_count, obs_per_group)
        print("      {} total groups required".format(num_groups))

        # The required number of nodes
        total_procs = num_groups * group_size

    else:
        print("      Using smallest possible group size.")
        # Using the max number of processes per node, compute the number of processes
        # needed for one observation based on memory requirements.
        # This is the initial guess of the group size.
        if obs_mem > node_mem:
            # One observation requires multiple nodes.  Round up to the nearest whole
            # node.
            print(
                "      Max observation mem > node mem, using one or more nodes per group"
            )
            group_size = node_procs * (1 + int(obs_mem / node_mem))
        else:
            # One observation requires less than a node.  Try to fit multiple
            # observations on one node.
            print(
                "      Max observation mem <= node mem, using up to one node per group"
            )
            group_mem = node_mem
            group_size = node_procs
            while (group_size % 2 == 0) and (group_mem > (2 * obs_mem)):
                group_size = group_size // 2
                group_mem = group_mem / 2
        print(
            "      Minimum group size = {} procs, {} procs per node".format(
                group_size, node_procs
            )
        )

        if ceil_div(group_size, node_procs) > max_nodes:
            print(
                "      Required nodes for one group ({}) is greater than max ({}), skipping job".format(
                    ceil_div(group_size, node_procs), max_nodes
                )
            )
            return (None, None, None, None)

        # If the node count (at the max processes per node) to achieve the
        # required memory results in too many processes for the observation,
        # Reduce the process count and increase threading as needed while
        # maintaining the number of nodes for memory requirements.

        while group_size > obs_max_procs:
            print(
                "      Group size ({}) exceeds observation limit ({}).  Adjusting to:".format(
                    group_size, obs_max_procs
                )
            )
            node_procs = node_procs // 2
            omp_threads *= 2
            group_size = group_size // 2
            print(
                "        {} procs per node ({} threads per proc), group size = {}".format(
                    node_procs, omp_threads, group_size
                )
            )

        # We now have the size of one group, assuming that each group has one
        # observation.  Compute the number of nodes needed.
        total_procs = obs_count * group_size

    nodes = ceil_div(total_procs, node_procs)
    if nodes > max_nodes:
        print(
            "      Required nodes ({}) is greater than max ({}), skipping job".format(
                nodes, max_nodes
            )
        )
        return (None, None, None, None)

    while node_procs > total_procs:
        # Our job is smaller than one node, reduce the size
        node_procs = node_procs // 2
        omp_threads *= 2

    print(
        "      Using {} nodes with {} procs per node, {} threads per proc, and {} procs per group".format(
            nodes, node_procs, omp_threads, group_size
        )
    )

    return (nodes, node_procs, omp_threads, group_size)


def main():
    parser = argparse.ArgumentParser(description="Generate TOAST example scripts.")

    parser.add_argument(
        "--config",
        required=False,
        default="config.toml",
        help="The input config file in TOML format",
    )

    parser.add_argument(
        "--datadir",
        required=False,
        default=None,
        help="The directory where you have fetched the data",
    )

    parser.add_argument(
        "--large_groups",
        required=False,
        default=False,
        action="store_true",
        help="Use largest groups possible, rather than smallest.",
    )

    # Location of this script
    example_dir = os.path.dirname(os.path.realpath(__file__))

    args = parser.parse_args()

    datadir = args.datadir
    if datadir is None:
        datadir = os.path.join(example_dir, "data")

    config = toml.load(args.config)

    systems = list(config["systems"].keys())

    jobtypes = list(config["jobs"].keys())

    for system in systems:
        print("Generating scripts for {}:".format(system))
        sysprops = config["systems"][system]
        template = sysprops["template"]
        _, template_ext = os.path.splitext(template)

        for job in jobtypes:
            job_props = config["jobs"][job]
            print("  Job type {}:".format(job))
            sizes = [x for x in job_props.keys() if x != "common"]

            # Parse common parameters
            common = job_props["common"]
            common_tele = common["telescopes"]
            common_pipeline = common["pipeline"]

            for sz in sizes:
                # The properties for this job size
                size_props = job_props[sz]

                print("    size {}:".format(sz))
                jobname = "{}_{}".format(job, sz)
                outdir = "job_{}_{}_{}".format(system, job, sz)
                os.makedirs(outdir, exist_ok=True)

                # link to data directory
                jobdata = os.path.join(outdir, "data")
                if os.path.islink(jobdata):
                    # a previously created link...
                    os.remove(jobdata)
                os.symlink(datadir, jobdata)

                # link scripts
                # src_py = os.path.join(example_dir, "trigger_astropy.py")
                # dest_py = os.path.join(outdir, "trigger_astropy.py")
                # if os.path.islink(dest_py):
                #     os.remove(dest_py)
                # os.symlink(src_py, dest_py)

                src_py = os.path.join(example_dir, "pysm_sky.py")
                dest_py = os.path.join(outdir, "pysm_sky.py")
                if os.path.islink(dest_py):
                    os.remove(dest_py)
                os.symlink(src_py, dest_py)

                telescopes = dict()
                focalplanes = list()
                schedules = list()
                for ctele, stele in zip(common_tele, size_props["telescopes"]):
                    tname = ctele["name"]
                    if tname != stele["name"]:
                        raise RuntimeError(
                            "common and per-size telescopes must have same names and order"
                        )
                    fp = dict(ctele["focalplane"])
                    if "focalplane" in stele:
                        fp.update(stele["focalplane"])
                    schd = dict()
                    if "schedule" in ctele:
                        schd.update(ctele["schedule"])
                    if "schedule" in stele:
                        schd.update(stele["schedule"])
                    telescopes[tname] = dict()
                    telescopes[tname]["focalplane"] = fp
                    telescopes[tname]["schedule"] = schd
                    focalplanes.append(
                        "focalplane_{}_{}.pkl".format(tname, fp["minpix"])
                    )
                    if len(schd) > 0:
                        schedules.append("schedule_{}.txt".format(tname))

                # Write pipeline parameter file
                pipeline = dict(common_pipeline)
                if "pipeline" in size_props:
                    pipeline.update(size_props["pipeline"])
                pipeline_file = os.path.join(outdir, "pipeline.par")
                create_argparse_file(pipeline, pipeline_file)

                # Compute the job size
                req = size_props["requirements"]

                (nodes, node_procs, omp_threads, group_size) = compute_job_size(
                    sysprops["max_nodes"],
                    sysprops["node_mem_gb"],
                    sysprops["node_max_procs"],
                    sysprops["omp_min_threads"],
                    req["obs_count"],
                    req["obs_mem_gb"],
                    req["obs_max_procs"],
                    large_groups=args.large_groups,
                )
                if nodes is None:
                    continue

                # Build up the job substitution dictionary.  Substitute our calculated
                # node parameters.

                jobsub = dict(sysprops)
                jobsub.update(req)
                jobsub["node_procs"] = node_procs
                jobsub["omp_threads"] = omp_threads
                jobsub["group_size"] = group_size
                jobsub["nodes"] = nodes
                jobsub["jobname"] = jobname
                jobsub["focalplane_list"] = "--focalplane {}".format(
                    ",".join(focalplanes)
                )
                jobsub["schedule_list"] = ""
                if len(schedules) > 0:
                    jobsub["schedule_list"] = "--schedule {}".format(
                        ",".join(schedules)
                    )

                pysm_fwhm = None
                pysm_bandcenter = None
                pysm_bandwidth = None
                for tele, tprops in telescopes.items():
                    fp = tprops["focalplane"]
                    if pysm_fwhm is None:
                        pysm_fwhm = fp["fwhm"]
                        pysm_bandcenter = fp["bandcenter_ghz"]
                        pysm_bandwidth = fp["bandwidth_ghz"]
                    else:
                        if pysm_fwhm != fp["fwhm"]:
                            raise RuntimeError(
                                "These jobs only support a single det type"
                            )
                        if pysm_bandcenter != fp["bandcenter_ghz"]:
                            raise RuntimeError(
                                "These jobs only support a single det type"
                            )
                        if pysm_bandwidth != fp["bandwidth_ghz"]:
                            raise RuntimeError(
                                "These jobs only support a single det type"
                            )

                # Read the template and write the job script
                outfile = os.path.join(outdir, "run{}".format(template_ext))
                with open(outfile, "w") as fout:
                    with open(template, "r") as fin:
                        for line in fin:
                            if re.match("@TELESCOPES@.*", line) is not None:
                                # Write our commands to create telescopes
                                line = ""
                                for tele, tprops in telescopes.items():
                                    tcom = create_telescope_commands(
                                        tele,
                                        tprops["focalplane"],
                                        tprops["schedule"],
                                        outdir,
                                        sysprops["mpi_launch"],
                                        sysprops["mpi_procs"],
                                        sysprops["mpi_nodes"],
                                    )
                                    line = "{}{}".format(line, tcom)
                            elif re.match("@PYSM_SKY@.*", line) is not None:
                                line = create_pysm_commands(
                                    pipeline["input-map"],
                                    pipeline["nside"],
                                    pysm_bandcenter,
                                    pysm_bandwidth,
                                    pysm_fwhm,
                                    pipeline["coord"],
                                    sysprops["mpi_launch"],
                                    sysprops["mpi_procs"],
                                    sysprops["mpi_nodes"],
                                )
                            else:
                                for k, v in jobsub.items():
                                    line = re.sub(
                                        "@{}@".format(k), "{}".format(v), line
                                    )
                            fout.write(line)
                if template_ext == ".sh":
                    st = os.stat(outfile)
                    os.chmod(outfile, st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    main()
