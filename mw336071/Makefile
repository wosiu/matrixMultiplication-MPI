CC = mpicc
CXX = mpic++
CFLAGS  = -Wall -c -O3
CXXFLAGS = -Wall -c --std=c++11
LDFLAGS = -Wall --std=c++11
MATGENFILE = densematgen.o
ALL = matrixmul matrixmul_deb
DEPEN_H = utils.h
DEPEN = utils.o
DEPEN_DEB = utils.o

all: $(ALL)

matrixmul: matrixmul.o $(MATGENFILE) $(DEPEN)
	$(CXX) $(LDFLAGS) -O3 $^ -o $@

matrixmul_deb: matrixmul_deb.o $(MATGENFILE) $(DEPEN_DEB)
	$(CXX) $(LDFLAGS) -g -ggdb $^ -o $@

matrixmul.o: matrixmul.cpp densematgen.h $(DEPEN_H) Makefile
	$(CXX) $(CXXFLAGS) -O3 -o $@ $< 

matrixmul_deb.o: matrixmul.cpp densematgen.h $(DEPEN_H) Makefile
	$(CXX) $(CXXFLAGS) -g -ggdb -o $@ $< 

utils.o: utils.cpp Makefile
	$(CXX) $(CXXFLAGS) -O3 -o $@ $< 	

$(MATGENFILE): densematgen.c Makefile
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL) 
