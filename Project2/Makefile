
default: main.exe

device.o: device.cuh device.cu
	nvcc device.cu

preprocessing.o: preprocessing.cuh preprocessing.cu
	nvcc preprocessing.cu

main.exe: device.o preprocessing.o main.cuh main.cu
	nvcc main.cu device.o preprocessing.o -o main.exe