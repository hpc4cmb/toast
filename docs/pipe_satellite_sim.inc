.. code-block:: console

    def main():
        env = Environment.get()
        log = Logger.get()

        mpiworld, procs, rank, comm = pipeline_tools.get_comm()
        args, comm, groupsize = parse_arguments(comm, procs)

        # Parse options

        if comm.world_rank == 0:
            os.makedirs(args.outdir, exist_ok=True)

        focalplane, gain, detweights = load_focalplane(args, comm)

        data = create_observations(args, comm, focalplane, groupsize)

        pipeline_tools.expand_pointing(args, comm, data)

        signalname = None
        skyname = pipeline_tools.simulate_sky_signal(
            args, comm, data, [focalplane], "signal"
        )
        if skyname is not None:
            signalname = skyname

        skyname = pipeline_tools.apply_conviqt(args, comm, data, "signal")
        if skyname is not None:
            signalname = skyname

        diponame = pipeline_tools.simulate_dipole(args, comm, data, "signal")
        if diponame is not None:
            signalname = diponame

        # Mapmaking.

        if not args.use_madam:
            if comm.world_rank == 0:
                log.info("Not using Madam, will only make a binned map")

            npp, zmap = pipeline_tools.init_binner(args, comm, data, detweights)

            # Loop over Monte Carlos

            firstmc = args.MC_start
            nmc = args.MC_count

            for mc in range(firstmc, firstmc + nmc):
                outpath = os.path.join(args.outdir, "mc_{:03d}".format(mc))

                pipeline_tools.simulate_noise(
                    args, comm, data, mc, "tot_signal", overwrite=True
                )

                # add sky signal
                pipeline_tools.add_signal(args, comm, data, "tot_signal", signalname)

                if gain is not None:
                    op_apply_gain = OpApplyGain(gain, name="tot_signal")
                    op_apply_gain.exec(data)

                if mc == firstmc:
                    # For the first realization, optionally export the
                    # timestream data.  If we had observation intervals defined,
                    # we could pass "use_interval=True" to the export operators,
                    # which would ensure breaks in the exported data at
                    # acceptable places.
                    pipeline_tools.output_tidas(args, comm, data, "tot_signal")
                    pipeline_tools.output_spt3g(args, comm, data, "tot_signal")

                pipeline_tools.apply_binner(
                    args, comm, data, npp, zmap, detweights, outpath, "tot_signal"
                )

        else:

            # Initialize madam parameters

            madampars = pipeline_tools.setup_madam(args)

            # Loop over Monte Carlos

            firstmc = args.MC_start
            nmc = args.MC_count

            for mc in range(firstmc, firstmc + nmc):
                # create output directory for this realization
                outpath = os.path.join(args.outdir, "mc_{:03d}".format(mc))

                pipeline_tools.simulate_noise(
                    args, comm, data, mc, "tot_signal", overwrite=True
                )

                # add sky signal
                pipeline_tools.add_signal(args, comm, data, "tot_signal", signalname)

                if gain is not None:
                    op_apply_gain = OpApplyGain(gain, name="tot_signal")
                    op_apply_gain.exec(data)

                pipeline_tools.apply_madam(
                    args, comm, data, madampars, outpath, detweights, "tot_signal"
                )

                if comm.comm_world is not None:
                    comm.comm_world.barrier()
