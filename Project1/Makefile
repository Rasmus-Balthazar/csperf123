HEADERS = data_gen.h
default: independent

datagen: data_gen.c $(HEADERS)
	gcc data_gen.c -o data_gen.o

independent: independent.c $(HEADERS)
	gcc independent.c -o independent.o