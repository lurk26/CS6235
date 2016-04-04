
CC=/usr/local/cuda/bin/nvcc -std=c++11
INCLUDE=-I/usr/local/cuda/include
HEADER=-I/usr/local/cuda/samples/common/inc
FLAGS = -g -Wall
SOURCES = dns.cpp

EXECUTABLES = dns

all:
	$(CC) $(HEADER) $(CFLAS) $(SOURCES) -o $(EXECUTABLES)
	./$(EXECUTABLES)
clean:
        
	rm -rf dns.o
