
INCLUDEDIRS := -I.
INCLUDEDIRS += -I$(shell pg_config --includedir-server)
INCLUDEDIRS += -I$(shell pg_config --includedir)
INCLUDEDIRS += -I/home/buote/programming/praktikum/include/
# If you are using shared libraries, make sure this location can be
# found at runtime (see /etc/ld.so.conf and ldconfig command).
LIBDIR = -L$(shell pg_config --libdir)
# This is where the shared object should be installed
LIBINSTALL = $(shell pg_config --pkglibdir)

hd_gm.so: gm.c Makefile
			gcc -fpic -o hd_gm.o -c hd_gm.c $(INCLUDEDIRS)
			gcc -shared -o hd_gm.so hd_gm.o $(LIBDIR) 
			sudo cp hd_gm.so $(LIBINSTALL)

hd_hmm.so: hd_hmm.c Makefile
			gcc -fpic -o hd_hmm.o -c hd_hmm.c $(INCLUDEDIRS)
			gcc -shared -o hd_hmm.so hd_hmm.o $(LIBDIR) 
			sudo cp hd_hmm.so $(LIBINSTALL)

install: all
	sudo cp hd_gm.so $(LIBINSTALL)
	sudo cp hd_hmm.so $(LIBINSTALL)


.PHONY: all
.DEFAULT: all
all: hd_gm.so hd_hmm.so
