import sys
import numpy as np
import numpy.ctypeslib as npc

import quaternionarray as qa
import time
import ctypes as ct
from ctypes.util import find_library
from ctypes import c_void_p

c_qarray = ct.CDLL("./benchmark/libqarray.so")

n_loop = 100#0000
n_data = 1000000#   

print("~~~ c_qarray benchmarking ~~~")
print("Array length = ", n_data)
print("Looping ", n_loop, " times")

q = np.empty((n_data,4))
for i in range(0,4):
	q[:,i] = i + 2.0
qa.norm_inplace(q)
q2 = q
timearr = np.array([0., 9] )	
targettime = np.array([0., 3, 4.5, 9])
vec = np.array([ 0.57734543,  0.30271255,  0.75831218])
axis = np.empty((n_data,3))
for i in range(0,3):
	axis[:,i] = i + 2.0
qa.norm_inplace(axis)
angle = np.random.randn(n_data)

print("======= Rotation =======")
print("Python library runtime: ")
chrono_rot_py = time.clock()
for i in range(1,n_loop):
	qa.rotate(q,vec)
chrono_rot_py = time.clock() - chrono_rot_py
print(chrono_rot_py, "s")
print("---")

print("C library runtime: ")
chrono_rot_c = time.clock()
vec_out = np.empty_like(q[:,:3])
for i in range(1,n_loop):
	c_qarray.qrotate(int(n_data), vec.ctypes.data_as(c_void_p), q.ctypes.data_as(c_void_p), vec_out.ctypes.data_as(c_void_p))
chrono_rot_c = time.clock() - chrono_rot_c
print(chrono_rot_c, "s")

speed_up = chrono_rot_py/chrono_rot_c
print('x%.1f times faster' % speed_up)
print("---")

print("(with argtypes)")
c_qarray.qrotate.argtypes = [
    ct.c_int,
    npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')]
chrono_rot_carg = time.clock()
vec_out = np.empty_like(q[:,:3])
for i in range(1,n_loop):
	c_qarray.qrotate(n_data, vec, q, vec_out)
chrono_rot_carg = time.clock() - chrono_rot_carg
print(chrono_rot_carg, "s")

slow_down = 100*(chrono_rot_carg-chrono_rot_c)/chrono_rot_c
print('%.0f%% slower (w/ vs w/o argtypes)' % slow_down)


print("======= Multiplication =======")
print("Python library runtime: ")
chrono_mult_py = time.clock()
for i in range(1,n_loop):
	qa.mult(q, q2)
chrono_mult_py = time.clock() - chrono_mult_py
print(chrono_mult_py, "s")
print("---")

print("C library runtime: ")
chrono_mult_c = time.clock()
r_out = np.empty_like(q)
for i in range(1,n_loop):
	c_qarray.qmult(int(n_data), q.ctypes.data_as(c_void_p), q2.ctypes.data_as(c_void_p), r_out.ctypes.data_as(c_void_p))
chrono_mult_c = time.clock() - chrono_mult_c
print(chrono_mult_c, "s")

speed_up = chrono_mult_py/chrono_mult_c
print('x%.1f times faster' % speed_up)
print("---")

print("(with argtypes)")
c_qarray.qmult.argtypes = [
    ct.c_int,
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')]
chrono_mult_cargs = time.clock()
r_out = np.empty_like(q)
for i in range(1,n_loop):
	c_qarray.qmult(n_data, q, q2, r_out)
chrono_mult_cargs = time.clock() - chrono_mult_cargs
print(chrono_mult_cargs, "s")

slow_down = 100*(chrono_mult_cargs-chrono_mult_c)/chrono_mult_c
print('%.0f%% slower (w/ vs w/o argtypes)' % slow_down)



print("======= From axis and angle =======")
print("Python library runtime: ")
chrono_faa_py = time.clock()
for i in range(1,n_loop):
	qa.rotation(axis, angle)
chrono_faa_py = time.clock() - chrono_faa_py
print(chrono_faa_py, "s")
print("---")

print("C library runtime: ")
chrono_faa_c = time.clock()
q_out = np.empty_like(q)
for i in range(1,n_loop):
	c_qarray.from_axisangle(int(n_data), axis.ctypes.data_as(c_void_p), angle.ctypes.data_as(c_void_p), q_out.ctypes.data_as(c_void_p))
chrono_faa_c = time.clock() - chrono_faa_c
print(chrono_faa_c, "s")

speed_up = chrono_faa_py/chrono_faa_c
print('x%.1f times faster' % speed_up)
print("---")

print("(with argtypes)")
c_qarray.from_axisangle.argtypes = [
    ct.c_int,
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    npc.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')]
chrono_faa_cargs = time.clock()
q_out = np.empty_like(q)
for i in range(1,n_loop):
	c_qarray.from_axisangle(n_data, axis, angle, q_out)
chrono_faa_cargs = time.clock() - chrono_faa_cargs
print(chrono_faa_cargs, "s")

slow_down = 100*(chrono_faa_cargs-chrono_faa_c)/chrono_faa_c
print('%.0f%% slower (w/ vs w/o argtypes)' % slow_down)



