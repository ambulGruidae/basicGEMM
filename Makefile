all:
	nvcc ./main.cu -lcublas -o gemm