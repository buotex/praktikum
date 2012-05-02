PGINCLUDES = `pg_config --includedir-server`
CFLAGS     = -fpic -fPIC -frounding-math -I.
LDFLAGS    = -larmadillo
# If you are using shared libraries, make sure this location can be
# found at runtime (see /etc/ld.so.conf and ldconfig command).
CXXFLAGS = $(CFLAGS) -std=c++0x
CFLAGS += -Ihmm -I$(PGINCLUDES)

LIBDIR = -L$(shell pg_config --libdir)
OBJ = hd_gm.o hd_hmm.o libhmm.o
# This is where the shared object should be installed
LIBINSTALL = $(shell pg_config --pkglibdir)
.PHONY: all
.DEFAULT: all
all: hd_gm.o hd_hmm.o libhmm.o hd_hmm.so
install: all
		sudo cp hd_hmm.so $(LIBINSTALL)

hd_gm.o: hd_gm.c Makefile
		cc $(CFLAGS) -o hd_gm.o -c hd_gm.c 

hd_hmm.o: hd_hmm.c Makefile
		cc $(CFLAGS) -o hd_hmm.o -c hd_hmm.c 

libhmm.o: hmm/kld.cpp Makefile
		c++ $(CXXFLAGS) -o libhmm.o -c hmm/kld.cpp

hd_hmm.so: $(OBJ)
		c++ -shared $(OBJ) ${LDFLAGS}

clean:
	rm *.o *.so

