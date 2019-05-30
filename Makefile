#

# Override Mac compiler to use OpenMP
#CXX		= g++-8
CXX		= icc
#CXXFLAGS	= -std=c++11 -Wall -g -O3 -fopenmp -pthread -m64 -arch x86_64 -I${MKLROOT}/include
CXXFLAGS	= -std=c++11 -Wall -g -O3 -qopenmp -pthread -m64 -qopt-assume-safe-padding -xAVX -axCore-AVX512 -qopt-zmm-usage=high -qopt-report=5 -I${MKLROOT}/include

#CXXFLAGS	 ?= -std=c++11 -Wall -g -O3 -qopenmp -qopt-assume-safe-padding -qopt-report=5 -xAVX
CXXFLAGS+=$(BUILD)

SRCDIRS = ./src/
SRCFILES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/MPCC.cpp))
SRCS = MPCC.cpp $(SRCFILES) 
OBJS = $(SRCFILES:%.cpp=%.o)

LIBDIR		= -L$(MKLROOT)/lib
LIB 		= -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lpthread -lstdc++ -lm -ldl
#LIB 		= -lmkl_intel_lp64 -lmkl_core -liomp5 -lpthread -lstdc++ -lm -ldl
#LIB 		= -mkl -liomp5 -lpthread -lstdc++ -lm -ldl
LIBS		 = $(LIBDIR) $(LIB)

IDIR =../include

ODIR=./

$(ODIR)/%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

MPCC: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f ./*.o *~ ./src/*.o 

