
CXX := g++
CXXFLAGS := -O0 -g -fPIC -I/home/kisner/software/conda/envs/toast/include

flac_test : main.cpp
	$(CXX) $(CXXFLAGS) -o flac_test main.cpp -lFLAC -lm

clean :
	rm -f flac_test *.o

