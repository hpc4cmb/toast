/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by 
a BSD-style license that can be found in the LICENSE file.
*/


#include <Python.h>
#include <ctoast.h>


/* datatypes */

typedef enum {
    item_type_int8,
    item_type_uint8,
    item_type_int16,
    item_type_uint16,
    item_type_int32,
    item_type_uint32,
    item_type_int64,
    item_type_uint64,
    item_type_float32,
    item_type_float64,
    item_type_string
} ToastItemType;


/* Define the ToastBuffer object structure */

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go below. */
    size_t size;
    size_t itemsize;
    ToastItemType itype;
    void * data;
} ToastBuffer;


/* This is the __init__ function, implemented in C */
static int ToastBuffer_init ( ToastBuffer * self, PyObject * args, PyObject * kwds ) {

    /* init may have already been called */
    if ( self->data != NULL ) {
        ctoast_mem_aligned_free ( self->data );
    }

    /* unpack the size and data type from the keywords */

    static char *kwlist[] = {"size", "type"};

    size_t typelen;
    char const * typestr;

    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "ks", kwlist, &(self->size), typestr) ) {
        return -1;
    }

    if ( self->size < 0 ) {
        self->size = 0;
    }

    if ( typestr == NULL ) {
        PyErr_SetString(PyExc_ValueError, "Data type string is NULL");
        return -1;
    } else {
        typelen = strlen ( typestr );

        if ( typelen == strlen("float64") ) {
            if ( strncmp ( typestr, "float64", typelen ) == 0 ) {
                self->itype = item_type_float64;
                self->itemsize = sizeof(double);
            }
        }
        
        if ( typelen == strlen("float32") ) {
            if ( strncmp ( typestr, "float32", typelen ) == 0 ) {
                self->itype = item_type_float32;
                self->itemsize = sizeof(float);
            }
        }
        
        if ( typelen == strlen("int64") ) {
            if ( strncmp ( typestr, "int64", typelen ) == 0 ) {
                self->itype = item_type_int64;
                self->itemsize = sizeof(int64_t);
            }
        }
        
        if ( typelen == strlen("uint64") ) {
            if ( strncmp ( typestr, "uint64", typelen ) == 0 ) {
                self->itype = item_type_uint64;
                self->itemsize = sizeof(uint64_t);
            }
        }

        if ( typelen == strlen("int32") ) {
            if ( strncmp ( typestr, "int32", typelen ) == 0 ) {
                self->itype = item_type_int32;
                self->itemsize = sizeof(int32_t);
            }
        }
        
        if ( typelen == strlen("uint32") ) {
            if ( strncmp ( typestr, "uint32", typelen ) == 0 ) {
                self->itype = item_type_uint32;
                self->itemsize = sizeof(uint32_t);
            }
        }

        if ( typelen == strlen("int16") ) {
            if ( strncmp ( typestr, "int16", typelen ) == 0 ) {
                self->itype = item_type_int16;
                self->itemsize = sizeof(int16_t);
            }
        }
        
        if ( typelen == strlen("uint16") ) {
            if ( strncmp ( typestr, "uint16", typelen ) == 0 ) {
                self->itype = item_type_uint16;
                self->itemsize = sizeof(uint16_t);
            }
        }

        if ( typelen == strlen("int8") ) {
            if ( strncmp ( typestr, "int8", typelen ) == 0 ) {
                self->itype = item_type_int8;
                self->itemsize = sizeof(int8_t);
            }
        }
        
        if ( typelen == strlen("uint8") ) {
            if ( strncmp ( typestr, "uint8", typelen ) == 0 ) {
                self->itype = item_type_uint8;
                self->itemsize = sizeof(uint8_t);
            }
        }

        if ( typelen == strlen("string") ) {
            if ( strncmp ( typestr, "string", typelen ) == 0 ) {
                self->itype = item_type_string;
                self->itemsize = sizeof(char);
            }
        }

        PyErr_SetString(PyExc_ValueError, "Invalid data type");
        return -1;
    }

    self->data = ctoast_mem_aligned_alloc ( self->size * self->itemsize );

    return 0;
}


/* this function is called when the object is deallocated */

static void ToastBuffer_dealloc ( ToastBuffer * self ) {
    if ( self->data != NULL ) {
        ctoast_mem_aligned_free ( self->data );
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}


/* This function returns the string representation of our object */

static PyObject * ToastBuffer_str ( ToastBuffer * self ) {
    char const * s = "test";


    PyObject* ret = PyUnicode_FromString(s);
    //free(s);
    return ret;
}


/* Here is the buffer interface function */

static int ToastBuffer_getbuffer ( PyObject *obj, Py_buffer *view, int flags ) {
    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    ToastBuffer * self = (ToastBuffer*)obj;

    view->obj = (PyObject*)self;
    
    view->buf = (void*)self->data;
    view->len = self->size * self->itemsize;
    view->readonly = 0;
    view->itemsize = self->itemsize;

    switch ( self->itype ) {
        case item_type_float64 :
            view->format = "d";
            break;
        case item_type_float32 :
            view->format = "f";
            break;
        case item_type_int64 :
            view->format = "q";
            break;
        case item_type_uint64 :
            view->format = "Q";
            break;
        case item_type_int32 :
            view->format = "i";
            break;
        case item_type_uint32 :
            view->format = "I";
            break;
        case item_type_int16 :
            view->format = "h";
            break;
        case item_type_uint16 :
            view->format = "H";
            break;
        case item_type_int8 :
            view->format = "b";
            break;
        case item_type_uint8 :
            view->format = "B";
            break;
        case item_type_string :
            view->format = "c";
            break;
        default :
            PyErr_SetString(PyExc_ValueError, "Invalid data type");
            return -1;
            break;
    }
    
    view->ndim = 1;    
    view->shape = &self->size;  // length-1 sequence of dimensions
    view->strides = &view->itemsize;  // for the simple case we can do this
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(self);  // need to increase the reference count
    return 0;
}


static PyBufferProcs ToastBuffer_as_buffer = {
    /* this definition is only compatible with Python 3.3 and above. */
    (getbufferproc)ToastBuffer_getbuffer,
    (releasebufferproc)0,  /* we do not require any special release function */
};


/* 
Here is the type structure: we put the above functions in the appropriate 
place in order to actually define the Python object type.
*/

static PyTypeObject ToastBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cbuffer.ToastBuffer",        /* tp_name */
    sizeof(ToastBuffer),            /* tp_basicsize */
    0,                            /* tp_itemsize */
    (destructor)ToastBuffer_dealloc,/* tp_dealloc */
    0,                            /* tp_print */
    0,                            /* tp_getattr */
    0,                            /* tp_setattr */
    0,                            /* tp_reserved */
    (reprfunc)ToastBuffer_str,      /* tp_repr */
    0,                            /* tp_as_number */
    0,                            /* tp_as_sequence */
    0,                            /* tp_as_mapping */
    0,                            /* tp_hash  */
    0,                            /* tp_call */
    (reprfunc)ToastBuffer_str,      /* tp_str */
    0,                            /* tp_getattro */
    0,                            /* tp_setattro */
    &ToastBuffer_as_buffer,         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,           /* tp_flags */
    "ToastBuffer object",           /* tp_doc */
    0,                            /* tp_traverse */
    0,                            /* tp_clear */
    0,                            /* tp_richcompare */
    0,                            /* tp_weaklistoffset */
    0,                            /* tp_iter */
    0,                            /* tp_iternext */
    0,                            /* tp_methods */
    0,                            /* tp_members */
    0,                            /* tp_getset */
    0,                            /* tp_base */
    0,                            /* tp_dict */
    0,                            /* tp_descr_get */
    0,                            /* tp_descr_set */
    0,                            /* tp_dictoffset */
    (initproc)ToastBuffer_init,     /* tp_init */
};


/* now we initialize the Python module which contains our new object: */

static PyModuleDef cbuffer_module = {
    PyModuleDef_HEAD_INIT,
    "cbuffer",
    "Extension type for ToastBuffer object.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit_cbuffer(void) 
{
    PyObject* m;

    ToastBufferType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ToastBufferType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&cbuffer_module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&ToastBufferType);
    PyModule_AddObject(m, "ToastBuffer", (PyObject *)&ToastBufferType);
    return m;
}

