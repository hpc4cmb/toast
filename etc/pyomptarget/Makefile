
# GCC 12.1.0 with offload support
#CXX := g++-12
#CXXFLAGS := -O0 -g -fPIC -pthread -std=c++11 -fcf-protection=none -fno-stack-protector
#CXX_OMP_FLAGS := -fopenmp -foffload=nvptx-none='-Wa,-m,sm_80 -misa=sm_80 -fPIC -lm -latomic'

# NVHPC
# CXX = nvc++
# CXXFLAGS = -O0 -g -fPIC -pthread -std=c++11
# CXX_OMP_FLAGS = -mp=gpu -Minfo=mp -gpu=cc86

# AMD LLVM
CXX := amdclang++
CXXFLAGS := -O3 -g -fPIC -std=c++11
CXX_OMP_FLAGS := -fPIC -fopenmp -fopenmp-target-debug=3 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030
CXX_LD_FLAGS := -lomp -lomptarget -lomptarget.rtl.amdgpu

pybind := ../../src/toast/pybind11
modext := $(shell python3-config --extension-suffix)
pyincl := $(shell python3-config --includes)


all : device_info test pyomptarget$(modext)


pyomptarget$(modext) : module.o
	$(CXX) -shared $(CXX_OMP_FLAGS) -o pyomptarget$(modext) module.o $(CXX_LD_FLAGS)

module.o : module.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OMP_FLAGS) -I. -I$(pybind)/include $(pyincl) -c -o module.o module.cpp

device_info : device_info.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OMP_FLAGS) -o device_info device_info.cpp $(CXX_LD_FLAGS)

test : test.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OMP_FLAGS) -o test test.cpp $(CXX_LD_FLAGS)

clean :
	rm -f *.so *.o device_info test

