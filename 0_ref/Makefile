CXX=icpc
CXXFLAGS=-xHost -O3 
#CXXFLAGS+=-qopt-zmm-usage=high
#CXXFLAGS+=-qopt-multiple-gather-scatter-by-shuffles
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
