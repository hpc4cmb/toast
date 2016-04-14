
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

np.import_array()

ctypedef np.float64_t f64_t
ctypedef np.float64_t f32_t
ctypedef np.int64_t i64_t
ctypedef np.uint64_t u64_t
ctypedef np.int32_t i32_t
ctypedef np.uint32_t u32_t
ctypedef np.int16_t i16_t
ctypedef np.uint16_t u16_t
ctypedef np.int8_t i8_t
ctypedef np.uint8_t u8_t

cdef extern from "pytoast.h":
    double * pytoast_mem_aligned_f64(size_t n)
    float * pytoast_mem_aligned_f32(size_t n)
    i64_t * pytoast_mem_aligned_i64(size_t n)
    u64_t * pytoast_mem_aligned_u64(size_t n)
    i32_t * pytoast_mem_aligned_i32(size_t n)
    u32_t * pytoast_mem_aligned_u32(size_t n)
    i16_t * pytoast_mem_aligned_i16(size_t n)
    u16_t * pytoast_mem_aligned_u16(size_t n)
    i8_t * pytoast_mem_aligned_i8(size_t n)
    u8_t * pytoast_mem_aligned_u8(size_t n)
    void pytoast_mem_aligned_free(void * mem)


cdef _alloc_f64(u64_t size):
    cdef np.npy_intp npsize = size
    cdef f64_t * mem
    cdef np.ndarray[f64_t, ndim=1] ret
    mem = <f64_t*>pytoast_mem_aligned_f64(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_FLOAT64, <void*>mem)
    return ret

cdef _alloc_f32(u64_t size):
    cdef np.npy_intp npsize = size
    cdef f32_t * mem
    cdef np.ndarray[f32_t, ndim=1] ret
    mem = <f32_t*>pytoast_mem_aligned_f32(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_FLOAT32, <void*>mem)
    return ret

cdef _alloc_i64(u64_t size):
    cdef np.npy_intp npsize = size
    cdef i64_t * mem
    cdef np.ndarray[i64_t, ndim=1] ret
    mem = pytoast_mem_aligned_i64(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_INT64, <void*>mem)
    return ret

cdef _alloc_u64(u64_t size):
    cdef np.npy_intp npsize = size
    cdef u64_t * mem
    cdef np.ndarray[u64_t, ndim=1] ret
    mem = pytoast_mem_aligned_u64(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_UINT64, <void*>mem)
    return ret

cdef _alloc_i32(u64_t size):
    cdef np.npy_intp npsize = size
    cdef i32_t * mem
    cdef np.ndarray[i32_t, ndim=1] ret
    mem = pytoast_mem_aligned_i32(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_INT32, <void*>mem)
    return ret

cdef _alloc_u32(u64_t size):
    cdef np.npy_intp npsize = size
    cdef u32_t * mem
    cdef np.ndarray[u32_t, ndim=1] ret
    mem = pytoast_mem_aligned_u32(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_UINT32, <void*>mem)
    return ret

cdef _alloc_i16(u64_t size):
    cdef np.npy_intp npsize = size
    cdef i16_t * mem
    cdef np.ndarray[i16_t, ndim=1] ret
    mem = pytoast_mem_aligned_i16(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_INT16, <void*>mem)
    return ret

cdef _alloc_u16(u64_t size):
    cdef np.npy_intp npsize = size
    cdef u16_t * mem
    cdef np.ndarray[u16_t, ndim=1] ret
    mem = pytoast_mem_aligned_u16(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_UINT16, <void*>mem)
    return ret

cdef _alloc_i8(u64_t size):
    cdef np.npy_intp npsize = size
    cdef i8_t * mem
    cdef np.ndarray[i8_t, ndim=1] ret
    mem = pytoast_mem_aligned_i8(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_INT8, <void*>mem)
    return ret

cdef _alloc_u8(u64_t size):
    cdef np.npy_intp npsize = size
    cdef u8_t * mem
    cdef np.ndarray[u8_t, ndim=1] ret
    mem = pytoast_mem_aligned_u8(size)
    ret = np.PyArray_SimpleNewFromData(1, &npsize, np.NPY_UINT8, <void*>mem)
    return ret


def _alloc(np.ndarray[u64_t, ndim=1] dims, dtype):
    cdef u64_t ndim = dims.shape[0]
    cdef u64_t nelem = 1

    for i in range(ndim):
        nelem = nelem * dims[i]

    data = None
    if dtype == np.float64:
        data = _alloc_f64(nelem)
    elif dtype == np.float32:
        data = _alloc_f32(nelem)
    elif dtype == np.int64:
        data = _alloc_i64(nelem)
    elif dtype == np.uint64:
        data = _alloc_u64(nelem)
    elif dtype == np.int32:
        data = _alloc_i32(nelem)
    elif dtype == np.uint32:
        data = _alloc_u32(nelem)
    elif dtype == np.int16:
        data = _alloc_i16(nelem)
    elif dtype == np.uint16:
        data = _alloc_u16(nelem)
    elif dtype == np.int8:
        data = _alloc_i8(nelem)
    elif dtype == np.uint8:
        data = _alloc_u8(nelem)
    else:
        raise RuntimeError("Unsupported data type")

    return data


def _free(np.ndarray data):
    cdef void * raw = np.PyArray_DATA(data)
    pytoast_mem_aligned_free(raw)
    return

