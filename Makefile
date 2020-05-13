#

# Override Mac compiler to use OpenMP
#CXX		= g++-8
CXX		= ${CC}
#CXXFLAGS	= -std=c++11 -Wall -g -O3 -fopenmp -pthread -m64 -I${MKLROOT}/include ${OPENBLAS_INC} ${CUDA_INC}
CXXFLAGS	= -std=c++11 -Wall -g -O3 -qopenmp -pthread -m64 -qopt-assume-safe-padding -xAVX -axCore-AVX512 -qopt-zmm-usage=high -qopt-report=5 -I${MKLROOT}/include ${OPENBLAS_INC} ${CUDA_INC}
#CXXFLAGS	= -std=c++11 -Wall -g -O3 -qopenmp -pthread -m64 -qopt-assume-safe-padding -xAVX -axCore-AVX512 -qopt-zmm-usage=high -qopt-report=5 -I${MKLROOT}/include ${R_INC}

#CXXFLAGS	 ?= -std=c++11 -Wall -g -O3 -qopenmp -qopt-assume-safe-padding -qopt-report=5 -xAVX
CXXFLAGS+=$(BUILD)

SRCDIRS = ./src
SRCFILES = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/MPCC.cpp))
SRCS = MPCC.cpp MPCCnaive.cpp $(SRCFILES) 
OBJS = $(SRCFILES:%.cpp=%.o)

#LIBDIR		= -L$(MKLROOT)/lib
LIBDIR		= -L$(MKLROOT)/lib ${OPENBLAS_LIB} ${CUDA_LIB} ${R_LIB}
#LIB 		= -DMKL -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lpthread -lstdc++ -lm -ldl
LIB 		= -DMKL -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lopenblas -lcuda -lpthread -lstdc++ -lm -ldl
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

