
default: main.exe

device.o: device.cuh device.cu
	nvcc -c device.cu

preprocessing.o: preprocessing.cuh preprocessing.cu
	nvcc -c preprocessing.cu

main.exe: device.o preprocessing.o main.cuh main.cu
	nvcc main.cu device.o preprocessing.o -o main.exe