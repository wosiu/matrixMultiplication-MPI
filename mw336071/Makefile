CC = mpicc
CXX = mpic++
CFLAGS  = -Wall -c -O3
CXXFLAGS = -Wall -c --std=c++11 #-O3
LDFLAGS = -Wall --std=c++11 # -O3
MATGENFILE = densematgen.o
ALL = matrixmul matrixmul_deb
 
all: $(ALL)

matrixmul: matrixmul.o $(MATGENFILE)
	$(CXX) $(LDFLAGS) -O3 $^ -o $@

matrixmul_deb: matrixmul_deb.o $(MATGENFILE)
	$(CXX) $(LDFLAGS) -g $^ -o $@

matrixmul.o: matrixmul.cpp densematgen.h Makefile
	$(CXX) $(CXXFLAGS) -O3 -o $@ $< 

matrixmul_deb.o: matrixmul.cpp densematgen.h Makefile
	$(CXX) $(CXXFLAGS) -g -o $@ $< 

$(MATGENFILE): densematgen.c Makefile
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL) 
