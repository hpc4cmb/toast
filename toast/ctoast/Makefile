
# This Makefile is not used for production, and exists only for
# implementing stand-alone, low-level tests of compiler options
# and other features which are challenging to profile / debug
# from within python.

CC = gcc
CFLAGS = -O3 -fPIC -I.
LIBS = -lm

HEADERS = pytoast.h

LIBOBJ = pytoast_mem.o pytoast_qarray.o

all : test_mem test_qarray


test_mem : test_mem.o libctoast.a
	$(CC) -o test_mem test_mem.o libctoast.a $(LIBS)

test_qarray : test_qarray.o libctoast.a
	$(CC) -o test_qarray test_qarray.o libctoast.a $(LIBS)

libctoast.a : $(LIBOBJ)
	ar rcs libctoast.a $(LIBOBJ)

%.o : %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

clean :
	@rm -f test_mem test_qarray *.a *.o *~
