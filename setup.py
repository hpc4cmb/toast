# This setup.py file simply builds TOAST using the underlying cmake build
# system.  This is only preferred in certain cases where the automation is
# easier from a setup.py (e.g. readthedocs, pip, etc).

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        # Assuming Makefiles
        build_args += ["--", "-j2"]

        self.build_args = build_args

        env = os.environ.copy()
        env["CXXFLAGS"] = "{} -DVERSION_INFO=\\'{}\\'".format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.check_call(
            ["cmake", cmake_list_dir] + cmake_args, cwd=self.build_temp, env=env
        )

        print("-" * 10, "Building extensions", "-" * 40)
        cmake_cmd = ["cmake", "--build", "."] + self.build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        extpath = self.get_ext_filename(ext.name)
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = os.path.join(build_temp, "src", extpath)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


ext_modules = [CMakeExtension("toast._libtoast")]

try:
    from mpi4py import MPI

    # If we can import mpi4py, then assume that the MPI extension
    # will be built.
    ext_modules.append(CMakeExtension("toast._libtoast_mpi"))
except ImportError:
    pass

version = None
with open("RELEASE", "r") as rel:
    version = rel.readline().rstrip()

conf = dict()
conf["name"] = "toast"
conf["description"] = "Time Ordered Astrophysics Scalable Tools"
conf["author"] = "Theodore Kisner, Reijo Keskitalo"
conf["author_email"] = "tskisner.public@gmail.com"
conf["license"] = "BSD"
conf["url"] = "https://github.com/hpc4cmb/toast"
conf["version"] = version
conf["provides"] = "toast"
conf["python_requires"] = ">=3.4.0"
conf["install_requires"] = ["cmake", "numpy", "scipy", "healpy", "matplotlib"]
conf["packages"] = find_packages("src")
conf["package_dir"] = {"": "src"}
conf["ext_modules"] = ext_modules
conf["cmdclass"] = {"build_ext": CMakeBuild}
conf["zip_safe"] = False

setup(**conf)
