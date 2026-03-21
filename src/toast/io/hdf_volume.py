# Copyright (c) 2026-2026 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import re
import numbers
from collections.abc import MutableMapping
import tempfile

import numpy as np
from astropy import units as u

from ..coordinates import azel_to_radec
from ..mpi import MPI
from ..instrument import GroundSite
from ..observation import default_values as defaults
from .. import qarray as qa
from ..timing import function_timer, Timer
from ..utils import Logger, sqlite_connect, sqlite_scalar
from .observation_hdf_load import load_hdf5


class VolumeIndex(object):
    """Class representing per-observation metadata in a volume.

    Args:
        path (str):  The path to the volume sqlite DB.

    """

    default_name = "index.sqlite"

    obs_table = "observations"

    def __init__(self, path):
        self._path = path
        self._reset_db_info()

    def _reset_db_info(self):
        # These are only relevant on the rank zero process.
        self._db_schema = None
        self._db_null_count = None
        self._db_rows = -1

    @function_timer
    def select(self, query, comm=None):
        """Run a query on the index DB and return the native SQL results.

        This executes the query and fetches all results (a list of tuples).  If comm
        is not None, the query is run on the rank 0 process and the results broadcast
        across the communicator.

        Args:
            query (str):  The SQL query.
            comm (MPI.Comm):  The optional MPI communicator

        Returns:
            (list):  The list of result tuples.

        """
        log = Logger.get()
        result = None
        if comm is None or comm.rank == 0:
            # Some columns may have NULL values.  We want to warn the user
            # that their query may be over a subset of the data.
            self._load_db_schema()
            warn = f"Selection: '{query}' uses columns with NULL counts:"
            do_warn = False
            for col_name, n_null in self._db_null_count.items():
                if re.search(col_name, query) is not None and n_null > 0:
                    do_warn = True
                    warn += f" {col_name} ({n_null}/{self._db_rows}),"
            if do_warn:
                log.warning(warn[:-1])

            conn = sqlite_connect(self._path, mode="r")
            cur = conn.cursor()
            cur.execute(query)
            result = cur.fetchall()
            cur.close()
            del cur
            conn.close()
            del conn
        if comm is not None:
            result = comm.bcast(result, root=0)
        return result

    def _parse_index_fields(self, obs, indexfields):
        """Parse strings into actual object references in the observation."""
        attr_pat = re.compile(r"\.(\w*)(.*)")
        dict_pat = re.compile(r"\[(\w*)\](.*)")
        log = Logger.get()

        def _op_reduce(input, op):
            if op == "mean":
                return np.mean(input)
            elif op == "median":
                return np.median(input)
            elif op == "first":
                return input[0]
            elif op == "last":
                return input[-1]
            else:
                msg = f"Invalid reduction operation '{op}'"
                raise RuntimeError(msg)

        def _get_obj(parent, key):
            if parent is None or key == "":
                return parent
            attr_mat = attr_pat.match(key)
            if attr_mat is None:
                # Not an attribute, maybe a dictionary key?
                dict_mat = dict_pat.match(key)
                if dict_mat is None:
                    msg = f"VolumeIndex invalid key {key} for object {parent}"
                    raise RuntimeError(msg)
                else:
                    sub_key = dict_mat.group(1)
                    remainder = dict_mat.group(2)
                    # The key might be string, or it might be an integer
                    # position into a list or tuple.
                    if isinstance(parent, MutableMapping):
                        # This is dictionary-like.  Try a string key.
                        try:
                            child = parent[sub_key]
                        except KeyError:
                            # Try converting key to int.
                            try:
                                elem = int(sub_key)
                                child = parent[elem]
                            except (ValueError, KeyError):
                                # Not an int...
                                msg = f"key {sub_key} not valid for object {parent}"
                                log.warning(msg)
                                child = None
                    else:
                        # Try converting key to integer index
                        try:
                            elem = int(sub_key)
                            child = parent[elem]
                        except (ValueError, KeyError):
                            # Not an int...
                            msg = f"key {sub_key} not valid for object {parent}"
                            log.warning(msg)
                            child = None
                    return _get_obj(child, remainder)
            else:
                # Parse the attribute
                sub_key = attr_mat.group(1)
                remainder = attr_mat.group(2)
                child = getattr(parent, sub_key)
                return _get_obj(child, remainder)

        props = dict()
        if indexfields is None:
            return props
        for fld, raw in indexfields.items():
            key_op = raw.split(":")
            obj = _get_obj(obs, key_op[0])
            if obj is None:
                # This observation has no value for the field
                val = None
            elif len(key_op) > 1:
                # We are doing a reduction, does that make sense?
                if not isinstance(obj, (np.ndarray, list, tuple)):
                    msg = f"Cannot apply reduction '{key_op[1]}' to object {obj}"
                    raise RuntimeError(msg)
                val = _op_reduce(obj, key_op[1])
            else:
                if isinstance(obj, (np.ndarray, list, tuple)):
                    # This is array-like, but no reduction was specified.
                    # Use the mean.
                    val = np.mean(obj)
                else:
                    # A scalar
                    val = obj
            props[fld] = val
        return props

    def _field_schema(self, fields):
        """Determine the sqlite type for each field."""
        ftypes = dict()
        for fname, fval in fields.items():
            if fval is None:
                ftypes[fname] = "NONE"
            elif isinstance(fval, numbers.Integral):
                ftypes[fname] = "INTEGER"
            elif isinstance(fval, u.Quantity):
                # We store the floating point value
                ftypes[fname] = "REAL"
            elif isinstance(fval, numbers.Number):
                # We already checked integers above, so this must be float
                ftypes[fname] = "REAL"
            else:
                # Must be a string
                ftypes[fname] = "TEXT"
        return ftypes

    def _load_db_schema(self):
        """Query the current DB schema and number of NULL values"""
        if self._db_schema is not None:
            # Already loaded
            return
        conn = sqlite_connect(self._path, mode="r")
        cur = conn.cursor()

        # Get schema with column types
        cur.execute(f"PRAGMA table_info({self.obs_table})")
        col_info = cur.fetchall()
        self._db_schema = dict()
        for col in col_info:
            self._db_schema[col[1]] = col[2]

        # Get the total number of rows
        cur.execute(f"SELECT COUNT(*) FROM {self.obs_table}")
        self._db_rows = int(cur.fetchall()[0][0])

        # Get the number of nulls per column
        self._db_null_count = dict()
        for col_name in self._db_schema.keys():
            cur.execute(
                f"SELECT COUNT(*) FROM {self.obs_table} WHERE {col_name} IS NULL"
            )
            self._db_null_count[col_name] = int(cur.fetchall()[0][0])

        cur.close()
        del cur
        conn.close()
        del conn

    def _db_create(self, schema):
        """Create a new observation table."""
        self._db_schema = dict()
        self._db_rows = 0
        self._db_null_count = dict()

        # FIXME:  Decide what columns we should index after evaluating practical
        # performance on real-world volumes.

        create_str = f"CREATE TABLE {self.obs_table} ("
        fcreate = list()
        for k, v in schema.items():
            if v is None:
                msg = f"DB field {k} has type NONE.  The index must be created with"
                msg += " an observation that has valid values for all fields."
                raise RuntimeError(msg)
            if k == "name":
                # Primary key
                fcreate.append(f"{k} {v} PRIMARY KEY")
            else:
                fcreate.append(f"{k} {v}")
            self._db_schema[k] = v
            self._db_null_count[k] = 0
        create_str += ", ".join(fcreate)
        create_str += ")"
        # We create the temp directory in the same filesystem as the final index, so
        # that os.rename() will always work as expected.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(self._path)) as tdir:
            temp_path = os.path.join(tdir, "index.sqlite")
            conn = sqlite_connect(temp_path, mode="w")
            cur = conn.cursor()
            cur.execute(create_str)
            conn.commit()
            cur.close()
            del cur
            conn.close()
            del conn
            if not os.path.isfile(self._path):
                os.rename(temp_path, self._path)

    def _ground_scan_center(self, obs):
        """If the observation is from a GroundSite, compute the scan center."""
        if isinstance(obs.telescope.site, GroundSite):
            # FIXME: Eventually handle arbitrary data distributions here
            if not obs.is_distributed_by_detector:
                msg = "Indexing of GroundSite data only supported for "
                msg += "detector-distributed data"
                raise RuntimeError(msg)
            result = None
            if obs.comm.group_rank == 0:
                bad = (
                    obs.shared[defaults.shared_flags].data
                    & defaults.shared_mask_invalid
                )
                good = np.logical_not(bad)
                mean_az = np.mean(np.unwrap(obs.shared[defaults.azimuth].data[good]))
                mean_el = np.mean(obs.shared[defaults.elevation].data[good])
                mean_time = np.mean(obs.shared[defaults.times].data[good])
                bore_azel = qa.from_lonlat_angles(
                    -np.array([mean_az]), np.array([mean_el]), np.zeros(1)
                )
                bore_radec = azel_to_radec(
                    obs.telescope.site,
                    np.array([mean_time]),
                    bore_azel,
                )
                ra, dec, _ = qa.to_lonlat_angles(bore_radec)
                result = (mean_az, mean_el, ra[0], dec[0])
            if obs.comm.comm_group is not None:
                result = obs.comm.comm_group.bcast(result, root=0)
            return result
        else:
            return None

    @function_timer
    def append(self, obs, rel_path, indexfields=None):
        """Append observation to the index while extracting metadata.

        This appends the specified observation to the index while extracting
        metadata fields.  Basic properties of the observation (number of samples,
        time range, etc) are always indexed.  Additional index fields can be
        specified as a dictionary with `indexfields`.

        Each entry in indexfields is a mapping from an observation key / attribute
        to a field in the index.  The observation meta data to be indexed can be a
        scalar or a list / array.  In the case of multiple values in the specified
        metadata, this can be reduced to a scalar for the index using an operation
        specified by "mean", "median", "first" or "last" strings.  The syntax for each
        entry in the indexfields dictionary for an observation metadata key and for
        arbitrary attributes is:

            [obs_key_name]:OP   or
            .attr_name:OP

        This attribute access (".") and dictionary key access ("[...]") can be nested.
        For example:

            .attr_name.sub_attr_name[some_key][other_key]:OP

        For example, to index an observation key representing the nominal elevation
        of a ground simulation and store it in the index under "elevation", you could
        do:

            indexfields={"elevation": "scan_el"}

        Similarly, to index the simulated PWV from a weather object in a ground
        simulation you could do:

            indexfields={
                "elevation": "scan_el",
                "PWV": ".telescope.site.weather.pwv",
            }

        As a further example, imagine you have an attribute of the observation called
        "hk" that acts as a dictionary of housekeeping data.  You want to index the
        mean focalplane temperature, which is an array stored in `hk["temperature"]`:

            indexfields={
                "fp_temp": ".hk[temperature]:mean",
            }

        Note that we leave off the string quotes in this syntax.

        Args:
            obs (Observation):  The observation to add to the index.
            rel_path (str):  The relative path of the observation within the volume.
            indexfields (dict):  Additional fields to index.

        Returns:
            (None)

        """
        log = Logger.get()

        # Gather the total number of valid detectors
        n_invalid = np.count_nonzero(
            [y & defaults.det_mask_invalid for x, y in obs.local_detector_flags.items()]
        )
        n_valid_local = len(obs.local_detectors) - n_invalid
        if obs.comm_col is not None:
            n_valid = obs.comm_col.allreduce(n_valid_local, op=MPI.SUM)
        else:
            n_valid = n_valid_local
        n_valid = int(n_valid)

        # Get ground scan center, if available
        timer = Timer()
        timer.start()
        ground_props = self._ground_scan_center(obs)
        log.debug_rank(
            f"{obs.name} append: compute scan center in",
            comm=obs.comm.comm_group,
            timer=timer,
        )

        if obs.comm.group_rank == 0:
            # Get the user-specified properties
            fields = self._parse_index_fields(obs, indexfields)

            # Add in fields that we always index
            fields["name"] = obs.name
            fields["path"] = rel_path
            fields["uid"] = obs.uid
            fields["session"] = obs.session.name
            fields["samples"] = obs.n_all_samples
            fields["valid_dets"] = n_valid
            fields["start"] = obs.session.start.timestamp()
            fields["end"] = obs.session.end.timestamp()

            if ground_props is not None:
                az_center, el_center, ra_center, dec_center = ground_props
                fields["az"] = np.degrees(az_center)
                fields["el"] = np.degrees(el_center)
                fields["ra"] = np.degrees(ra_center)
                fields["dec"] = np.degrees(dec_center)

            field_types = self._field_schema(fields)

            if os.path.isfile(self._path):
                # DB already exists, verify schema.  We allow observations (except for
                # the first observation added) to be missing some fields.  These will
                # be assigned NULL values in the DB.
                msg = f"VolumeIndex.append(): {self._path} exists, checking schema"
                log.verbose(msg)
                self._load_db_schema()
                errmsg = f"Index {self._path}: "
                errmsg += f"Observation {obs.name} meta schema ({field_types}) does "
                errmsg += f"not match existing database schema ({self._db_schema})"
                if len(field_types) != len(self._db_schema):
                    raise RuntimeError(errmsg)
                for fkey, fval in field_types.items():
                    if fkey not in self._db_schema:
                        raise RuntimeError(errmsg)
                    if fval != "NONE" and fval != self._db_schema[fkey]:
                        raise RuntimeError(errmsg)
            else:
                # Create DB
                msg = f"VolumeIndex.append(): creating {self._path} with schema"
                msg += f" {field_types}"
                log.verbose(msg)
                self._db_create(field_types)

            # Write entry
            names = list()
            wilds = list()
            vals = list()
            for k, v in fields.items():
                names.append(f"{k}")
                wilds.append("?")
                vals.append(sqlite_scalar(v))
                if v is None:
                    # Increment null count
                    self._db_null_count[k] += 1
            # Increment row count
            self._db_rows += 1

            vals = tuple(vals)
            insert_str = f"INSERT INTO {self.obs_table} ("
            insert_str += ", ".join(names)
            insert_str += ")"
            insert_str += " VALUES ("
            insert_str += ", ".join(wilds)
            insert_str += ")"
            conn = sqlite_connect(self._path, mode="w")
            cur = conn.cursor()
            msg = f"VolumeIndex.append(): {insert_str} <= {vals}, "
            log.verbose(msg)
            cur.execute(insert_str, vals)
            conn.commit()
            cur.close()
            del cur
            conn.close()
            del conn
        log.debug_rank(
            f"{obs.name} append: insert into DB in",
            comm=obs.comm.comm_group,
            timer=timer,
        )

    @function_timer
    def append_file(self, volume_path, rel_path, indexfields=None, toastcomm=None):
        """Load observation metadata from a file and append.

        See the docstring of `append()` for details about the indexfields syntax.

        Args:
            volume_path (str):  The path to the top-level volume.
            rel_path (str):  The relative path within the volume of the observation
                file to append.
            indexfields (dict):  Additional fields to index.

        Returns:
            (None)

        """
        log = Logger.get()
        # Load the observation with only metadata and shared fields
        # (no detector data).
        obs_path = os.path.join(volume_path, rel_path)
        timer = Timer()
        timer.start()
        try:
            obs = load_hdf5(
                obs_path,
                toastcomm,
                process_rows=None,
                meta=None,
                detdata=list(),
                shared=None,
                intervals=list(),
                detectors=None,
                force_serial=False,
            )
            log.debug_rank(
                f"{obs.name} loaded metadata and pointing in",
                comm=toastcomm.comm_group,
                timer=timer,
            )
            # Append it
            self.append(obs, rel_path, indexfields=indexfields)
        except Exception as e:
            msg = f"File '{obs_path}' was not loadable as an Observation, skipping."
            log.warning(msg)
            raise e

    @staticmethod
    def find_observations(volume_path, pattern_str=r".*\.h5"):
        """Recursively search volume for files matching a pattern.

        The basename of each file is matched against pattern_str.  A list of relative
        paths to each observation file is returned.

        Args:
            volume_path (str):  The path to the top-level volume.
            pattern_str (str):  The pattern compiled into a regex for matching against
                each observation file basename.

        Returns:
            (list):  The list of observation relative paths.

        """
        pattern = re.compile(pattern_str)
        obs_files = list()
        for root, dirs, files in os.walk(volume_path):
            for fname in files:
                if pattern.search(fname) is not None:
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, start=volume_path)
                    obs_files.append(rel_path)
        return list(sorted(obs_files))

    def reindex(self, volume_path, indexfields=None, toastcomm=None, overwrite=False):
        log = Logger.get()
        if toastcomm.ngroups > 1:
            msg = "Toast communicator has multiple process groups- "
            msg += f"refusing to re-index volume '{volume_path}'"
            log.error(msg)
            raise RuntimeError(msg)

        msg = f"Re-building index for Volume '{volume_path}'"
        log.info_rank(msg, toastcomm.comm_world)

        # Reset any schema info
        self._reset_db_info()

        if toastcomm.world_rank == 0:
            if os.path.isfile(self._path):
                if not overwrite:
                    msg = f"Index path {self._path} already exists and overwrite"
                    msg += "is not enabled."
                    log.error(msg)
                    raise RuntimeError(msg)
                else:
                    os.remove(self._path)

        # One process builds the list of files to try
        obs_files = None
        if toastcomm.world_rank == 0:
            log.debug(f"reindex: scanning directory {volume_path}")
            obs_files = self.find_observations(volume_path)
            log.debug(f"reindex: found {len(obs_files)} observation files")
        if toastcomm.comm_world is not None:
            obs_files = toastcomm.comm_world.bcast(obs_files, root=0)
        for fname in obs_files:
            self.append_file(
                volume_path, fname, indexfields=indexfields, toastcomm=toastcomm
            )

    def __repr__(self):
        val = f"<VolumeIndex file = '{self._path}'>"
        return val
