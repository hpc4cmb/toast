# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.
"""OpenCL platform tools.
"""
import os
import ctypes
import numpy as np

from ..utils import Environment, Logger
from .utils import have_opencl, aligned_to_dtype, find_source

if have_opencl:
    import pyopencl as cl
    from pyopencl.array import Array


class OpenCL:
    """Singleton class to manage OpenCL usage.

    This class provides a global interface to the underlying OpenCL platforms,
    compiled programs, and memory management.

    The default device type can be controlled with environment variables:

    TOAST_OPENCL_DEFAULT=<value>

    Where supported values are "CPU", "GPU", and "OCLGRIND".

    """

    instance = None
    log_prefix = "OpenCL: "

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance._initialized = False
        return cls.instance

    def __init__(self):
        if self._initialized:
            return
        log = Logger.get()
        if not have_opencl:
            log.error("pyopencl is not available!")
        self._platforms = cl.get_platforms()

        self._cpus = list()
        self._cpu_names = dict()
        self._cpu_index = dict()
        self._gpus = list()
        self._gpu_names = dict()
        self._gpu_index = dict()
        self._grind = list()
        self._grind_names = dict()
        self._grind_index = dict()

        # The paradigm we use for multiple platforms is:
        # - For the first platform with CPUs we index all CPU devices from there,
        #   in case they are duplicated in other platforms.
        # - We index all GPUs across all platforms.

        found_cpus = False
        cpu_offset = 0
        gpu_offset = 0
        grind_offset = 0
        for iplat, plat in enumerate(self._platforms):
            devices = plat.get_devices()
            if not found_cpus:
                use_cpus = True
            else:
                use_cpus = False
            for dev in devices:
                if dev.type == 2:
                    # This is a CPU
                    found_cpus = True
                    if use_cpus:
                        self._cpus.append(
                            {
                                "name": dev.name,
                                "device": dev,
                                "platform": plat,
                                "type": "cpu",
                                "index": cpu_offset,
                            }
                        )
                        self._cpu_index[dev.name] = cpu_offset
                        self._cpu_names[cpu_offset] = dev.name
                        cpu_offset += 1
                elif dev.type == 4:
                    # This is a GPU
                    self._gpus.append(
                        {
                            "name": dev.name,
                            "device": dev,
                            "platform": plat,
                            "type": "gpu",
                            "index": gpu_offset,
                        }
                    )
                    self._gpu_index[dev.name] = gpu_offset
                    self._gpu_names[gpu_offset] = dev.name
                    gpu_offset += 1
                else:
                    # This is something else (OCLGRIND)
                    self._grind.append(
                        {
                            "name": dev.name,
                            "device": dev,
                            "platform": plat,
                            "type": "oclgrind",
                            "index": grind_offset,
                        }
                    )
                    self._grind_index[dev.name] = grind_offset
                    self._grind_names[grind_offset] = dev.name
                    grind_offset += 1

        self._n_cpu = cpu_offset
        self._n_gpu = gpu_offset
        self._n_grind = grind_offset
        if self._n_gpu == 0:
            self._default_gpu = -1
        else:
            self._default_gpu = 0
        if self._n_cpu == 0:
            self._default_cpu = -1
        else:
            self._default_cpu = 0
        if self._n_grind == 0:
            self._default_grind = -1
        else:
            self._default_grind = 0

        # Create contexts, queues and memory managers.  We split these
        # lists of devices in case we want to configure things differently
        # in the future.
        for d in self._cpus:
            d["context"] = cl.Context(
                devices=[d["device"]],
                properties=[(cl.context_properties.PLATFORM, d["platform"])],
            )
            # Default queue
            d["queue"] = cl.CommandQueue(d["context"], d["device"])
            d["allocator"] = cl.tools.ImmediateAllocator(
                d["queue"], mem_flags=cl.mem_flags.READ_WRITE
            )
            d["mempool"] = cl.tools.MemoryPool(d["allocator"])
        for d in self._gpus:
            d["context"] = cl.Context(
                devices=[d["device"]],
                properties=[(cl.context_properties.PLATFORM, d["platform"])],
            )
            # Default queue
            d["queue"] = cl.CommandQueue(d["context"], d["device"])
            d["allocator"] = cl.tools.ImmediateAllocator(
                d["queue"], mem_flags=cl.mem_flags.READ_WRITE
            )
            d["mempool"] = cl.tools.MemoryPool(d["allocator"])
        for d in self._grind:
            d["context"] = cl.Context(
                devices=[d["device"]],
                properties=[(cl.context_properties.PLATFORM, d["platform"])],
            )
            # Default queue
            d["queue"] = cl.CommandQueue(d["context"], d["device"])
            d["allocator"] = cl.tools.ImmediateAllocator(
                d["queue"], mem_flags=cl.mem_flags.READ_WRITE
            )
            d["mempool"] = cl.tools.MemoryPool(d["allocator"])

        # Indexed by program name and then device
        self._programs = dict()

        # Indexed by host buffer address and then device
        self._buffers = dict()

        # User override of default device type
        if "TOAST_OPENCL_DEFAULT" in os.environ:
            self._default_dev_type = os.environ["TOAST_OPENCL_DEFAULT"]
            if self._default_dev_type not in ["CPU", "GPU", "OCLGRIND"]:
                msg = "Unknown user specified default device type "
                msg += "'{self._default_dev_type}'"
                raise RuntimeError(msg)
            self._default_dev_type = self._default_dev_type.lower()
        else:
            if self._n_gpu > 0:
                # We have some GPUs, that is probably the intended default
                self._default_dev_type = "gpu"
            else:
                self._default_dev_type = "cpu"

        self._initialized = True

        # Create "Null" buffers.  Often we need to pass an optional device
        # array to kernels, and if they are not going to be used we still want
        # to pass a valid pointer.  So we pre-create some fake buffers for that
        # purpose.
        self._mem_null_host = dict()
        self._mem_null_dev = dict()
        for dt in [
            np.dtype(np.uint8),
            np.dtype(np.int8),
            np.dtype(np.uint16),
            np.dtype(np.int16),
            np.dtype(np.uint32),
            np.dtype(np.int32),
            np.dtype(np.uint64),
            np.dtype(np.int64),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ]:
            self._mem_null_host[dt] = np.zeros(1, dtype=dt)
            self._mem_null_dev[dt] = dict()
            for dev_type in ["cpu", "gpu", "oclgrind"]:
                self._mem_null_dev[dt][dev_type] = dict()
            for idx, d in enumerate(self._cpus):
                self._mem_null_dev[dt]["cpu"][idx] = self.mem_to_device(
                    self._mem_null_host[dt],
                    name=f"NULL_{dt}",
                    device_type="cpu",
                    device_index=idx,
                    async_=False,
                )
            for idx, d in enumerate(self._gpus):
                self._mem_null_dev[dt]["gpu"][idx] = self.mem_to_device(
                    self._mem_null_host[dt],
                    name=f"NULL_{dt}",
                    device_type="gpu",
                    device_index=idx,
                    async_=False,
                )
            for idx, d in enumerate(self._grind):
                self._mem_null_dev[dt]["oclgrind"][idx] = self.mem_to_device(
                    self._mem_null_host[dt],
                    name=f"NULL_{dt}",
                    device_type="oclgrind",
                    device_index=idx,
                    async_=False,
                )

    def __del__(self):
        # Free buffers
        if hasattr(self, "_buffers"):
            for haddr, devs in self._buffers.items():
                for dname, (dbuf, dsize, bname) in devs.items():
                    dbuf.finish()
                    del dbuf
                devs.clear()
            self._buffers.clear()

        # Free kernels and programs
        if hasattr(self, "_programs"):
            self._programs.clear()

        # Free memory pools, queues and contexts
        if (
            hasattr(self, "_cpus")
            and hasattr(self, "_gpus")
            and hasattr(self, "_grind")
        ):
            for devs in [self._cpus, self._gpus, self._grind]:
                for d in devs:
                    if "mempool" in d:
                        d["mempool"].free_held()
                        d["mempool"].stop_holding()
                        del d["mempool"]
                    if "allocator" in d:
                        del d["allocator"]
                    if "queue" in d:
                        d["queue"].flush()
                        d["queue"].finish()
                        del d["queue"]

    def get_device(self, device_type=None, device_index=None, device_name=None):
        """Lookup a device by type and either index or name.

        If the name and index are not specified, the default device of the
        selected type is returned.

        Args:
            device_type (str):  "cpu" or "gpu" or "oclgrind"
            device_index (int):  The index within either the CPU or GPU list.
            device_name (str):  The specific (and usually very long) device name.

        Returns:
            (dict):  The selected device properties.

        """
        if device_name is not None and device_index is not None:
            msg = "At most, one of device_name or device_index may be specified"
            raise RuntimeError(msg)
        if device_type is None:
            device_type = self._default_dev_type
        if device_type == "cpu":
            if device_name is not None:
                if device_name not in self._cpu_index:
                    msg = f"CPU device '{device_name}' does not exist"
                    raise RuntimeError(msg)
                device_index = self._cpu_index[device_name]
            else:
                if device_index is None:
                    device_index = self._default_cpu
            return self._cpus[device_index]
        elif device_type == "gpu":
            if device_name is not None:
                if device_name not in self._gpu_index:
                    msg = f"GPU device '{device_name}' does not exist"
                    raise RuntimeError(msg)
                device_index = self._gpu_index[device_name]
            else:
                if device_index is None:
                    device_index = self._default_gpu
            return self._gpus[device_index]
        elif device_type == "oclgrind":
            if device_name is not None:
                if device_name not in self._grind_index:
                    msg = f"OCLGRIND device '{device_name}' does not exist"
                    raise RuntimeError(msg)
                device_index = self._grind_index[device_name]
            else:
                if device_index is None:
                    device_index = self._default_grind
            return self._grind[device_index]
        else:
            msg = f"Unknown device type '{device_type}'"
            raise RuntimeError(msg)

    @property
    def n_cpu(self):
        return self._n_cpu

    @property
    def n_gpu(self):
        return self._n_gpu

    @property
    def n_oclgrind(self):
        return self._n_grind

    @property
    def default_device_type(self):
        return self._default_dev_type

    @property
    def default_gpu_index(self):
        return self._default_gpu

    @property
    def default_cpu_index(self):
        return self._default_cpu

    @property
    def default_oclgrind_index(self):
        return self._default_grind

    def info(self):
        """Print information about the general status of the OpenCL layer."""
        env = Environment.get()
        level = env.log_level()

        msg = ""
        for idev, dev in enumerate(self._cpus):
            msg += f"{self.log_prefix} CPU {idev} ({dev['name']})\n"
            msg += f"{self.log_prefix}   platform {dev['platform']}\n"
        for idev, dev in enumerate(self._gpus):
            msg += f"{self.log_prefix} GPU {idev} ({dev['name']})\n"
            msg += f"{self.log_prefix}   platform {dev['platform']}\n"
        for idev, dev in enumerate(self._grind):
            msg += f"{self.log_prefix} OCLGRIND {idev} ({dev['name']})\n"
            msg += f"{self.log_prefix}   platform {dev['platform']}\n"
        msg += f"{self.log_prefix} Default CPU = {self._default_cpu}\n"
        msg += f"{self.log_prefix} Default GPU = {self._default_gpu}\n"
        msg += f"{self.log_prefix} Default OCLGRIND = {self._default_grind}\n"
        print(msg, flush=True)

    def set_default_gpu(self, device_name=None, device_index=None):
        if self._n_gpu == 0:
            msg = "Cannot set default GPU, none are detected!"
            raise RuntimeError(msg)
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type="gpu"
        )
        self._default_gpu = dev["index"]

    def set_default_cpu(self, device_name=None, device_index=None):
        if self._n_cpu == 0:
            msg = "Cannot set default CPU, none are detected!"
            raise RuntimeError(msg)
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type="cpu"
        )
        self._default_cpu = dev["index"]

    def set_default_oclgrind(self, device_name=None, device_index=None):
        if self._n_grind == 0:
            msg = "Cannot set default OCLGRIND, none are detected!"
            raise RuntimeError(msg)
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type="oclgrind"
        )
        self._default_grind = dev["index"]

    def assign_default_devices(self, node_procs, node_rank, disabled):
        if self.n_cpu > 0:
            # Our platforms support CPUs
            proc_per_cpu = node_procs // self._n_cpu
            if self._n_cpu * proc_per_cpu < node_procs:
                proc_per_cpu += 1
            target = node_rank // proc_per_cpu
            self.set_default_cpu(device_index=target)
        if self.n_gpu > 0:
            # Our platforms support GPUs
            proc_per_gpu = node_procs // self._n_gpu
            if self._n_gpu * proc_per_gpu < node_procs:
                proc_per_gpu += 1
            target = node_rank // proc_per_gpu
            self.set_default_gpu(device_index=target)
            env = Environment.get()
            env.set_acc(self.n_gpu, proc_per_gpu, target)
        if self.n_oclgrind > 0:
            # Our platforms support OCLGRIND fake devices
            proc_per_grind = node_procs // self._n_grind
            if self._n_grind * proc_per_grind < node_procs:
                proc_per_grind += 1
            target = node_rank // proc_per_grind
            self.set_default_oclgrind(device_index=target)

    def build_program(self, program_name, source):
        """Load the program source and build for all devices."""
        with open(source, "r") as f:
            clstr = f.read()
        self._programs[program_name] = dict()
        for d in self._cpus:
            dname = d["name"]
            self._programs[program_name][dname] = cl.Program(d["context"], clstr)
            self._programs[program_name][dname].build()
            build_status = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.STATUS
            )
            build_log = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.LOG
            )
        for d in self._gpus:
            dname = d["name"]
            self._programs[program_name][dname] = cl.Program(d["context"], clstr)
            try:
                self._programs[program_name][dname].build()
            except:
                pass
            build_status = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.STATUS
            )
            build_log = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.LOG
            )
        for d in self._grind:
            dname = d["name"]
            self._programs[program_name][dname] = cl.Program(d["context"], clstr)
            self._programs[program_name][dname].build()
            build_status = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.STATUS
            )
            build_log = self._programs[program_name][dname].get_build_info(
                d["device"], cl.program_build_info.LOG
            )

    def context(self, device_name=None, device_index=None, device_type=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type=device_type
        )
        return dev["context"]

    def queue(self, device_name=None, device_index=None, device_type=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type=device_type
        )
        return dev["queue"]

    def has_kernel(
        self,
        program_name,
        kernel_name,
        device_name=None,
        device_index=None,
        device_type=None,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type=device_type
        )
        if program_name not in self._programs:
            return False
        devname = dev["name"]
        if devname not in self._programs[program_name]:
            return False
        prog = self._programs[program_name][devname]
        if not hasattr(prog, kernel_name):
            return False
        return True

    def kernel(
        self,
        program_name,
        kernel_name,
        device_name=None,
        device_index=None,
        device_type=None,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        exists = self.has_kernel(
            program_name,
            kernel_name,
            device_name=device_name,
            device_index=device_index,
            device_type=device_type,
        )
        if not exists:
            msg = f"kernel {kernel_name} in program {program_name} does not exist"
            raise RuntimeError(msg)
        dev = self.get_device(
            device_name=device_name, device_index=device_index, device_type=device_type
        )
        devname = dev["name"]
        prog = self._programs[program_name][devname]
        return getattr(prog, kernel_name)

    def get_or_build_kernel(
        self,
        program_name,
        kernel_name,
        device_name=None,
        device_index=None,
        device_type=None,
        source=None,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        if not self.has_kernel(
            program_name,
            kernel_name,
            device_name=device_name,
            device_index=device_index,
            device_type=device_type,
        ):
            if source is None:
                # Look for a source file named after the program in our
                # common directory
                source = find_source(os.path.dirname(__file__), f"{program_name}.cl")
            self.build_program(program_name, source)
        return self.kernel(
            program_name,
            kernel_name,
            device_name=device_name,
            device_index=device_index,
            device_type=device_type,
        )

    def _mem_host_props(self, host_buffer):
        if hasattr(host_buffer, "address"):
            # This is a C-allocated buffer
            haddr = host_buffer.address()
            hsize = host_buffer.size()
            # These are always 1D
            hshape = (hsize,)
            htype = aligned_to_dtype(host_buffer)
            harray = host_buffer.array()
        else:
            haddr = host_buffer.ctypes.data
            hsize = host_buffer.size
            hshape = host_buffer.shape
            htype = host_buffer.dtype
            harray = host_buffer
        return haddr, hsize, hshape, htype, harray

    def _mem_check_props(self, haddr, hsize, hname, dbuf, dsize, dname, err=False):
        log = Logger.get()
        # if dname != hname:
        #     msg = f"Host buffer {haddr} has device name '{dname}' not '{hname}'"
        #     if err:
        #         raise RuntimeError(msg)
        #     else:
        #         log.warning(msg)
        if dsize != hsize:
            msg = f"Host buffer {haddr} has device size {dsize} not {hsize}"
            if err:
                raise RuntimeError(msg)
            else:
                log.warning(msg)

    def _mem_check_exists(self, haddr, hsize, name, dev, err=False, warn=False):
        log = Logger.get()
        if haddr not in self._buffers:
            msg = f"Host buffer {haddr} (name={name}) not registered"
            if err:
                raise RuntimeError(msg)
            elif warn:
                log.warning(msg)
            return (None, None)
        dprops = self._buffers[haddr]
        dev_name = dev["name"]
        if dev_name not in dprops:
            msg = f"Host buffer {haddr} (name={name}) not on device '{dev_name}'"
            if err:
                raise RuntimeError(msg)
            elif warn:
                log.warning(msg)
            return (None, None)
        dbuf, dsize, dname = dprops[dev_name]
        self._mem_check_props(haddr, hsize, name, dbuf, dsize, dname, err=err)
        return dbuf, dsize

    def mem_create(self, host_buffer, name=None, device_type=None, device_index=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        if haddr not in self._buffers:
            self._buffers[haddr] = dict()
        dprops = self._buffers[haddr]
        dev_name = dev["name"]
        if dev_name in dprops:
            dbuf, dsize, dname = dprops[dev_name]
            msg = f"Host buffer {haddr} already registered on device "
            msg += f"'{dev_name}' at address {dbuf.base_data} with size {dsize}"
            msg += f" and name {dname}"
            raise RuntimeError(msg)
        dbuf = Array(
            dev["queue"],
            hshape,
            htype,
            order="C",
            allocator=dev["mempool"],
        )
        self._buffers[haddr][dev_name] = (
            dbuf,
            hsize,
            name,
        )
        return dbuf

    def mem_to_device(
        self,
        host_buffer,
        name=None,
        device_type=None,
        device_index=None,
        async_=False,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        if haddr not in self._buffers:
            self._buffers[haddr] = dict()
        dprops = self._buffers[haddr]
        dev_name = dev["name"]
        if dev_name in dprops:
            # Check that the existing array has correct properties
            dbuf, dsize, dname = dprops[dev_name]
            self._mem_check_props(haddr, hsize, name, dbuf, dsize, dname, err=True)
        else:
            dbuf = Array(
                dev["queue"],
                hshape,
                htype,
                order="C",
                allocator=dev["mempool"],
            )
            self._buffers[haddr][dev_name] = (
                dbuf,
                hsize,
                name,
            )
        dbuf.set(harray, async_=async_)
        return dbuf

    def mem_remove(self, host_buffer, name=None, device_type=None, device_index=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev, warn=True)
        if dbuf is None:
            return
        dbuf.finish()
        del dbuf
        del self._buffers[haddr][dev["name"]]
        if len(self._buffers[haddr]) == 0:
            del self._buffers[haddr]

    def mem_update_device(
        self,
        host_buffer,
        name=None,
        device_type=None,
        device_index=None,
        async_=False,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev, err=True)
        dbuf.set(harray, async_=async_)
        return dbuf

    def mem_update_host(
        self,
        host_buffer,
        name=None,
        device_type=None,
        device_index=None,
        async_=False,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev, err=True)
        if async_:
            _, ev = dbuf.get_async(ary=harray)
        else:
            dbuf.get(ary=harray)
            ev = None
        return ev

    def mem_reset(
        self,
        host_buffer,
        name=None,
        device_type=None,
        device_index=None,
        wait_for=None,
    ):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev, err=True)
        dbuf.fill(0, wait_for=wait_for)

    def mem_present(self, host_buffer, name=None, device_type=None, device_index=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev)
        if dbuf is None:
            return False
        else:
            return True

    def mem(self, host_buffer, name=None, device_type=None, device_index=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        dbuf, dsize = self._mem_check_exists(haddr, hsize, name, dev, err=True)
        return dbuf

    def mem_null(self, host_buffer, device_type=None, device_index=None):
        if device_type is None:
            device_type = self._default_dev_type
        dev = self.get_device(device_type=device_type, device_index=device_index)
        haddr, hsize, hshape, htype, harray = self._mem_host_props(host_buffer)
        return self._mem_null_dev[htype][dev["type"]][dev["index"]]

    def mem_dump(self):
        msg = ""
        prefix = f"{self.log_prefix}MEM:"
        for haddr, devs in self._buffers.items():
            for dname, (dbuf, dsize, bname) in devs.items():
                if dname in self._cpu_index:
                    dstr = f"CPU[{self._cpu_index[dname]}]"
                elif dname in self._gpu_index:
                    dstr = f"GPU[{self._gpu_index[dname]}]"
                elif dname in self._grind_index:
                    dstr = f"OCLGRIND[{self._grind_index[dname]}]"
                msg += f"{prefix} H[{haddr}] -> {dstr} size={dsize} ({bname})\n"
        print(msg, flush=True)
