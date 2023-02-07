CXX=icpx
#CXXFLAGS=-xHost -O3 -qopenmp
CXXFLAGS=-xHost -O3 -fsycl 
CXXFLAGS+=-qopt-zmm-usage=high
CXXFLAGS+= -I$(VTUNE_PROFILER_DIR)/include -g
LDFLAGS= -L$(VTUNE_PROFILER_DIR)/lib64 -littnotify
ifdef test
  CXXFLAGS+= -DTEST
endif
ifdef pbc
  CXXFLAGS+= -DPBC
endif
ifdef cutoff
  CXXFLAGS+= -DCUTOFF
endif
lj: lj.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
