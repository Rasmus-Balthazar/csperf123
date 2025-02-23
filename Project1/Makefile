#CC = gcc
#CFLAGS = -Wall -Werror
#CFLAGS = -Wall
#HEADERS = data_gen.h independent.h
#OBJ = data_gen.o independent.o

# default: independent

# datagen: data_gen.o
# 	$(CC) $(OBJ) -o datagen

# independent: independent.o data_gen.o
# 	$(CC) independent.o data_gen.o -o independent

# data_gen.o: data_gen.c $(HEADERS)
# 	$(CC) $(CFLAGS) -c data_gen.c -o data_gen.o

# independent.o: independent.c $(HEADERS)
# 	$(CC) $(CFLAGS) -c independent.c -o independent.o

# clean:
# 	rm -f $(OBJ) datagen independent



default: main.exe

data_gen.o: data_gen.h data_gen.c
	gcc -c data_gen.c

independent.o: data_gen.h independent.h independent.c
	gcc -pthread -c independent.c

count_then_move.o: data_gen.h count_then_move.c
	gcc -pthread -c count_then_move.c

main.exe: data_gen.o independent.o count_then_move.o main.c
	gcc main.c -pthread data_gen.o independent.o count_then_move.o -o main.exe
