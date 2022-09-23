
# GCC 12.1.0 with offload support
CXX := g++-12
CXXFLAGS := -O0 -g -fPIC -pthread -std=c++11 -fcf-protection=none -fno-stack-protector
CXX_OMP_FLAGS := -fopenmp -foffload=nvptx-none='-Wa,-m,sm_80 -misa=sm_80 -fPIC -lm -latomic'

# NVHPC
# CXX = nvc++
# CXXFLAGS = -O0 -g -fPIC -pthread -std=c++11
# CXX_OMP_FLAGS = -mp=gpu -Minfo=mp -gpu=cc86


pybind := ../../src/toast/pybind11
modext := $(shell python3-config --extension-suffix)
pyincl := $(shell python3-config --includes)


pyomptarget$(modext) : module.o
	$(CXX) -shared -o pyomptarget$(modext) module.o $(CXX_OMP_FLAGS)

module.o : module.cpp
	$(CXX) $(CXXFLAGS) $(CXX_OMP_FLAGS) -I. -I$(pybind)/include $(pyincl) -c -o module.o module.cpp


clean :
	rm -f *.so *.o

