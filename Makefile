CC = clang

.PHONY: gpt2
gpt2: gpt2/model.c
	$(CC) gpt2/model.c -O3 -lm -o gpt2/run

.PHONY: gpt2fast
gpt2fast: gpt2/model.c
	$(CC) gpt2/model.c -Ofast -lm -o gpt2/run

.PHONY: gpt2cl
gpt2cl: gpt2/model_cl.c
	$(CC) gpt2/model_cl.c -lm -framework OpenCL -o gpt2/run
