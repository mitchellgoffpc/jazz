CC = gcc

.PHONY: run
run: model.c
	$(CC) model.c -O3 -lm -o model.a

.PHONY: runfast
runfast: model.c
	$(CC) model.c -Ofast -lm -o model.a
