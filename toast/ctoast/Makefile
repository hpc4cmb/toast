
# This Makefile is not used for production, and exists only for
# implementing stand-alone, low-level tests of compiler options
# and other features which are challenging to profile / debug
# from within python.

CC = gcc
CFLAGS = -O0 -g -fPIC -I.
LIBS = -lm

HEADERS = pytoast.h

LIBOBJ = pytoast_mem.o

all : test_ctoast


test_ctoast : test_ctoast.o libctoast.a
	$(CC) -o test_ctoast test_ctoast.o libctoast.a $(LIBS)

libctoast.a : $(LIBOBJ)
	ar rcs libctoast.a $(LIBOBJ)

%.o : %.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

clean :
	@rm -f test_ctoast *.a *.o *~
