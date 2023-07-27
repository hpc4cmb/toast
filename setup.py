# This setup.py file simply builds TOAST using the underlying cmake build
# system.  This is only preferred in certain cases where the automation is
# easier from a setup.py (e.g. readthedocs, pip, etc).

import os
import sys
import re
import subprocess
import glob
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def find_compilers():
    cc = None
    cxx = None
    # If we have mpi4py, then get the MPI compilers that were used to build that.
    # Then get the serial compilers used by the MPI wrappers.  Otherwise return
    # None and later let CMake guess the compilers.
    try:
        from mpi4py import MPI
        import mpi4py

        mpiconf = mpi4py.get_config()
        mpicc = mpiconf["mpicc"]
        mpicxx = mpiconf["mpicxx"]
        mpicc_com = None
        mpicxx_com = None
        try:
            mpicc_com = subprocess.check_output(
                "{} -show".format(mpicc), shell=True, universal_newlines=True
            )
        except subprocess.CalledProcessError:
            # Cannot run the MPI C compiler, give up
            raise ImportError
        try:
            mpicxx_com = subprocess.check_output(
                "{} -show".format(mpicxx), shell=True, universal_newlines=True
            )
        except subprocess.CalledProcessError:
            # Cannot run the MPI C++ compiler, give up
            raise ImportError
        # Extract the serial compilers
        cc = mpicc_com.split()[0]
        cxx = mpicxx_com.split()[0]

    except (ImportError, KeyError):
        pass

    return (cc, cxx)


def get_version():
    # Run the underlying cmake command that generates the version file, and then
    # parse that output.  This way setup.py is using the exact same version as the
    # (not yet built) compiled code.
    topdir = Path(__file__).resolve().parent
    ver = None
    try:
        version_dir = os.path.join(topdir, "src", "libtoast")
        # version_dir = os.path.join("src", "libtoast")
        subprocess.check_call("cmake -P version.cmake", shell=True, cwd=version_dir)
        version_cpp = os.path.join(version_dir, "version.cpp")
        git_ver = None
        rel_ver = None
        with open(version_cpp, "r") as f:
            for line in f:
                mat = re.match(r'.*GIT_VERSION = "(.*)".*', line)
                if mat is not None:
                    git_ver = mat.group(1)
                mat = re.match(r'.*RELEASE_VERSION = "(.*)".*', line)
                if mat is not None:
                    rel_ver = mat.group(1)
        if (
            "READTHEDOCS" in os.environ
            or "CI" in os.environ
            or "CIBUILDWHEEL" in os.environ
        ):
            # We are running inside build infrastructure that requires a PEP440 version
            # and may be using a shallow clone, so the git version is malformed.  Always
            # use the release version in this case.
            ver = rel_ver
        elif (git_ver is not None) and (git_ver != ""):
            ver = git_ver
        else:
            ver = rel_ver
    except subprocess.CalledProcessError:
        raise RuntimeError("Cannot generate version!")
    return ver


class CMakeExtension(Extension):
    """
    This overrides the built-in extension class and essentially does nothing,
    since all extensions are compiled in one go by the custom build_ext class.
    """

    def __init__(self, name, sources=[]):
        super().__init__(name=name, sources=sources)


class CMakeBuild(build_ext):
    """
    Builds the full package using CMake.
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if extension.name == "toast._libtoast":
                # We always build the serial extension, so we trigger the build
                # on that.  This function may be called multiple times, so we
                # only build if the final output files do not exist.
                extpath = self.get_ext_filename(extension.name)
                dest_path = Path(self.get_ext_fullpath(extension.name)).resolve()
                build_lib = Path(self.build_lib).resolve()
                lib_path = os.path.join(build_lib, extpath)
                build_temp = Path(self.build_temp).resolve()
                source_path = os.path.join(build_temp, "src", extpath)
                if not os.path.isfile(source_path):
                    # The extension needs to be built
                    print(
                        f"Compiling extension '{source_path}' with CMake",
                        flush=True,
                    )
                    self.build_cmake()
                else:
                    print(
                        f"Compiled extension '{source_path}' already exists",
                        flush=True,
                    )
                if not os.path.isfile(dest_path) or not os.path.isfile(lib_path):
                    # The extension needs to be copied into place
                    print(
                        f"Copying built extension to '{dest_path}' and '{lib_path}'",
                        flush=True,
                    )
                    self.move_output(extension)
                else:
                    print(
                        f"Copy of extensions already exist at '{dest_path}' and '{lib_path}'",
                        flush=True,
                    )
        super().run()

    def build_cmake(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        (cc, cxx) = find_compilers()

        # Make a copy of the environment so that we can modify it
        env = os.environ.copy()

        # Search the environment for any variables starting with "TOAST_BUILD_".
        # We extract these and convert them into cmake options.
        cmake_opts = dict()
        cpat = re.compile(r"TOAST_BUILD_(.*)")
        for k, v in env.items():
            mat = cpat.match(k)
            if mat is not None:
                cmake_opts[mat.group(1)] = v

        cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]
        cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        # Set compilers

        if "CMAKE_C_COMPILER" in cmake_opts:
            # Get these from the environment
            cc = cmake_opts["CMAKE_C_COMPILER"]
            print(
                f"C Compiler:  using serial compiler '{cc}' from TOAST_BUILD environment variables"
            )
            cmake_args += [f"-DCMAKE_C_COMPILER={cc}"]
            _ = cmake_opts.pop("CMAKE_C_COMPILER")
        elif cc is not None:
            # Use serial compilers that were used when building MPI
            print(
                f"C Compiler:  using serial compiler '{cc}' from installed mpi4py package"
            )
            cmake_args += ["-DCMAKE_C_COMPILER={}".format(cc)]
        else:
            # We just let cmake guess the compilers and hope for the best...
            print(f"C Compiler:  not specified, will use CMake to discover")
            pass

        if "CMAKE_CXX_COMPILER" in cmake_opts:
            # Get these from the environment
            cxx = cmake_opts["CMAKE_CXX_COMPILER"]
            print(
                f"C++ Compiler:  using serial compiler '{cxx}' from TOAST_BUILD environment variables"
            )
            cmake_args += [f"-DCMAKE_CXX_COMPILER={cxx}"]
            _ = cmake_opts.pop("CMAKE_CXX_COMPILER")
        elif cxx is not None:
            # Use serial compilers that were used when building MPI
            print(
                f"C++ Compiler:  using serial compiler '{cxx}' from installed mpi4py package"
            )
            cmake_args += ["-DCMAKE_CXX_COMPILER={}".format(cxx)]
        else:
            # We just let cmake guess the compilers and hope for the best...
            print(f"C++ Compiler:  not specified, will use CMake to discover")
            pass

        # Append any other TOAST_BUILD_ options to the cmake args
        for k, v in cmake_opts.items():
            cmake_args += ["-D{}={}".format(k, v)]

        # Assuming Makefiles
        build_args += ["--", "-j2"]

        self.build_args = build_args

        env["CXXFLAGS"] = "{} -DVERSION_INFO=\\'{}\\'".format(
            env.get("CXXFLAGS", ""), get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_com = ["cmake", cmake_list_dir] + cmake_args
        print("\n".join([f"   {x}" for x in cmake_com]), flush=True)
        subprocess.check_call(cmake_com, cwd=self.build_temp, env=env)

        print("-" * 10, "Building extensions", "-" * 40, flush=True)
        cmake_cmd = ["cmake", "--build", "."] + self.build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        # If we are running on readthedocs, prepare sphinx inputs
        if "READTHEDOCS" in os.environ:
            subprocess.check_call([os.path.join("docs", "setup_docs.sh")])

    def move_output(self, ext):
        extpath = self.get_ext_filename(ext.name)
        build_temp = Path(self.build_temp).resolve()
        build_lib = Path(self.build_lib).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = os.path.join(build_temp, "src", extpath)
        lib_path = os.path.join(build_lib, extpath)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)
        # Also copy from the temp to the lib directory so that
        # the --inplace option works
        os.makedirs(os.path.dirname(lib_path), exist_ok=True)
        self.copy_file(source_path, lib_path)


ext_modules = [CMakeExtension("toast._libtoast")]

scripts = glob.glob("workflows/*.py")


def readme():
    with open("README.md") as f:
        return f.read()


conf = dict()
conf["name"] = "toast"
conf["description"] = "Time Ordered Astrophysics Scalable Tools"
conf["long_description"] = readme()
conf["long_description_content_type"] = "text/markdown"
conf["author"] = "Theodore Kisner, Reijo Keskitalo"
conf["author_email"] = "tskisner.public@gmail.com"
conf["license"] = "BSD"
conf["url"] = "https://github.com/hpc4cmb/toast"
conf["version"] = get_version()
conf["python_requires"] = ">=3.8.0"
conf["setup_requires"] = (["wheel"],)
conf["install_requires"] = [
    "tomlkit",
    "traitlets>=5.0",
    "numpy",
    "scipy",
    "matplotlib",
    "psutil",
    "h5py",
    "pshmem>=0.2.10",
    "pyyaml",
    "astropy",
    "healpy",
    "ephem",
]
conf["extras_require"] = {
    "mpi": ["mpi4py>=3.0"],
    "totalconvolve": ["ducc0"],
}
conf["packages"] = find_packages(
    "src",
)
conf["package_dir"] = {"": "src"}
conf["include_package_data"] = True
conf["exclude_package_data"] = {
    "": ["*.h", "*.c", "*.cpp", "*.hpp"],
}
conf["ext_modules"] = ext_modules
conf["scripts"] = scripts
conf["entry_points"] = {
    "console_scripts": [
        "toast_env = toast.scripts.toast_env:main",
        "toast_fake_focalplane = toast.scripts.toast_fake_focalplane:main",
        "toast_ground_schedule = toast.scripts.toast_ground_schedule:main",
        "toast_satellite_schedule = toast.scripts.toast_satellite_schedule:main",
        "toast_benchmark_satellite = toast.scripts.toast_benchmark_satellite:main",
        "toast_benchmark_ground_setup = toast.scripts.toast_benchmark_ground_setup:main",
        "toast_benchmark_ground = toast.scripts.toast_benchmark_ground:main",
        "toast_mini = toast.scripts.toast_mini:main",
        "toast_healpix_convert = toast.scripts.toast_healpix_convert:main",
        "toast_healpix_coadd = toast.scripts.toast_healpix_coadd:main",
        "toast_hdf5_to_spt3g = toast.scripts.toast_hdf5_to_spt3g:main",
        "toast_timing_plot = toast.scripts.toast_timing_plot:main",
        "toast_obsmatrix_combine = toast.scripts.toast_obsmatrix_combine:main",
        "toast_obsmatrix_coadd = toast.scripts.toast_obsmatrix_coadd:main",
        "toast_config_verify = toast.scripts.toast_config_verify:main",
    ]
}
conf["cmdclass"] = {"build_ext": CMakeBuild}
conf["zip_safe"] = False
conf["classifiers"] = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Astronomy",
]

setup(**conf)
