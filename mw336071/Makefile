CC = mpicc
CXX = mpic++
CFLAGS  = -Wall -c -O3
CXXFLAGS = -Wall -c -O3 --std=c++11
LDFLAGS = -Wall -O3 --std=c++11
MATGENFILE = densematgen.o

all: matrixmul

matrixmul: matrixmul.o $(MATGENFILE)
	$(CXX) $(LDFLAGS) $^ -o $@

matrixmul.o: matrixmul.cpp densematgen.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< 

$(MATGENFILE): densematgen.c Makefile
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
