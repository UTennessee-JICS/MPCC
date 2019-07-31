## Override Mac compiler to use OpenMP
CXX		= gcc
#CXX		= icc

INCLUDE = -I${MKLROOT}/include
#CXXFLAGS	= -std=c++11 -Wall -g -O3 -fopenmp -pthread -m64 -mavx512f -mavx512cd -I${MKLROOT}/include
CXXFLAGS	= -std=c++11 -g -O3 -fopenmp -m64 -mavx512f -mavx512cd  
#CXXFLAGS	= -std=c++11 -Wall -g -O3 -qopenmp -pthread -m64 -xAVX -axCore-AVX512 -qopt-assume-safe-padding -qopt-zmm-usage=high -qopt-report=5
CXXFLAGS+=$(BUILD)

SRCDIRS = ./src
SRCFILES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/MPCC*.cpp))
OBJS = $(SRCFILES:%.cpp=%.o)
#$(info SRCFILES = ${SRCFILES})
$(info OBJS = ${OBJS})

LIBDIR		= -L$(MKLROOT)/lib ${R_LIB}
LIB 		= -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lpthread -lstdc++ -lm -ldl
LIBS		 = $(LIBDIR) $(LIB)

#IDIR =../include
ODIR=./src

$(ODIR)/%.o: %.c
	$(CXX) -c $@ $< $(CXXFLAGS) ${INCLUDE}

MPCC: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) ${INCLUDE} $(LIBS)

.PHONY: clean

clean:
	rm -f ./*.o ./*.optrpt *~ ./src/*.o ./src/*.optrpt


