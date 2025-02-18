CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3 -g -DNDEBUG -pthread
LDFLAGS=-lpthread -pthread

all: affinity no-affinity affinityperformanceexample

affinity: affinity.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

no-affinity: no-affinity.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

affinityperformanceexample: affinityperformanceexample.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
