TOP=$(shell pwd)

# Files name
SWAN=simswan

BUILDDIR=$(TOP)/WaveComp/build
OBJDIR=$(BUILDDIR)/objects
MODDIR=$(BUILDDIR)/modules
LIBDIR=$(BUILDDIR)/lib

# FC compiler to use
F90=mpif90

# F2PY compiler
F2PY = f2py --fcompiler=gfortran

# Fortran optimisation flags
FCFLAGS=-O3 -funroll-loops --param max-unroll-times=2 -cpp \
    -ffree-form -ftree-vectorize -ffast-math -lstdc++ -ffree-line-length-none
FFLAGS= $(FCFLAGS) -I$(MODDIR)

F2PY_FLAGS= --f90exec=${F90} -I$(MODDIR)

LDFLAGS= -L$(LIBDIR)/

DIRMODS=WaveComp

.PHONY : all dist plugin dust clobber

all: dist

dist:
	@echo
	@echo "*************************************************"
	@echo "Ocean Plugin Makefile"
	@echo "*************************************************"
	@echo
	for i in $(DIRMODS) ; do   \
    	  ( cd $$i ; make dist) ;       \
	done
	@echo "*************************************************"
	@echo
	@echo "Build Swan binary."
	@echo
	@echo "*************************************************"
	@$(if $(wildcard SimSwan.so),rm -f *.so,)
	make libs

libs :	${SWAN}.so

${SWAN}.so: ${SWAN}.f90
	${F2PY} ${F2PY_FLAGS} $(LDFLAGS) -lWaveParam -c -m ${SWAN} $< 

%.o : %.f90
	$(F90) $(FFLAGS) -c $< -o $@

dust :
	for i in $(DIRMODS) ; do   \
    	( cd $$i ; make dust) ;       \
	done
	$(foreach module,$(MODULES),cd $(module); cd - ; )
	rm -fv *~ *.bak *.o *.mod *.original

clobber : dust
	for i in $(DIRMODS) ; do   \
    	( cd $$i ; make clobber) ;   \
	done
	rm -rfv $(BUILDDIR) *~ *.bak *.o *.mod *.original *.so
