# Copyright (c) 2015-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

CXX := g++
CXXFLAGS := -O0 -g -fPIC -I/home/kisner/software/conda/envs/toast/include

flac_test : main.cpp
	$(CXX) $(CXXFLAGS) -o flac_test main.cpp -lFLAC -lm

clean :
	rm -f flac_test *.o

