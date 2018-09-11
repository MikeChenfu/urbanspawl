export GDAL_HOME := /lustre/medusa/cgisgra/lcg_GOL/gdal
export CUDA_HOME := /sw/keeneland/cuda/4.1/linux_binary

LIB = -lgdal -ldl -fopenmp -lgomp  -L$(GDAL_HOME)/lib -lz
CLIB = -L$(CUDA_HOME)/lib64 -lcudart
FLAG = -I$(GDAL_HOME)/include/
CFLAG = -I$(CUDA_HOME)/include/

CC = mpicc
cc =nvcc

sleuth : main.o kernel.o
	 $(CC) main.o kernel.o -o sleuth $(FLAG)  $(LIB) $(CFLAG) $(CLIB)

main.o: main.cpp
	$(CC) -c main.cpp -o main.o $(FLAG)   $(LIB)

kernel.o: kernel.cu
	$(cc) -c kernel.cu -o  kernel.o $(CFLAG) $(CLIB)

clean:
	rm *.o*

