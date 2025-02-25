#CFLAGS = -Wall -Werror
#CFLAGS = -Wall

default: main.exe

data_gen.o: data_gen.h data_gen.c
	gcc -c data_gen.c

independent.o: data_gen.h independent.h independent.c
	gcc -pthread -c independent.c

count_then_move.o: data_gen.h count_then_move.h count_then_move.c
	gcc -pthread -c count_then_move.c

main.exe: data_gen.o independent.o count_then_move.o main.c main.h
	gcc main.c -pthread data_gen.o independent.o count_then_move.o -o main.exe
