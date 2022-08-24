#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <cublas_v2.h>

using namespace std;

#define BLOCK_NUM 128
#define THREAD_PER_BLOCK 32
#define dtype double // Set calculation Accuracy to double

struct my_timer
{
    struct timeval start_time, end_time;
    double time_use;

    void start()
    {
        gettimeofday(&start_time, NULL);
    }

    void stop()
    {
        gettimeofday(&end_time, NULL);
        time_use = (end_time.tv_sec - start_time.tv_sec) * 1.0e6 + end_time.tv_usec - start_time.tv_usec;
    }
};

void generate_matrix(dtype *&mat, int nnz)
{
    for (int i = 0; i < nnz; i++)
    {
        dtype x = rand() % 1000000;
        if (x < 1000000.0)
        {
            mat[i] = x / 10000.0 + 1.0;
        }
    }
}

void gemm_cublas(const dtype *A, const dtype *B, dtype *C, int m, int k, int n)
{

    dtype *d_A, *d_B, *d_C;
    size_t size = m * k * sizeof(dtype);
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    size = k * n * sizeof(dtype);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    size = m * n * sizeof(dtype);
    cudaMalloc(&d_C, size);
    cublasHandle_t s;
    dtype al = 1, ve = 0;
    cublasCreate_v2(&s);
    cublasDgemm(s, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &al, d_B, n, d_A, k, &ve, d_C, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
bool verify(const dtype *A, const dtype *B, dtype *C, int m, int k, int n)
{
    dtype *cpu_matC = (dtype *)malloc(m * n * sizeof(dtype));
    memset(cpu_matC, 0, m * n);
    gemm_cublas(A, B, cpu_matC, m, k, n);
    bool equal = true;
    for (int i = 0; i < m * n; ++i)
    {
        dtype fab_diff = fabs(cpu_matC[i] - C[i]);
        dtype fab_mat = fabs(cpu_matC[i]);
        if ((fab_mat > 1.0) && (fab_diff >= fab_mat * 1e-9))
        {
            equal = false;
        }
        else if ((fab_mat <= 1.0) && (fab_diff >= 1e-9))
        {
            equal = false;
        }

        if (!equal)
        {
            printf("check[%d]: %.5f != %.5f", i, cpu_matC[i], C[i]);
            // cout << "check: " << cpu_matC[i] << " != " << C[i] << endl;
            return false;
        }
    }

    free(cpu_matC);
    return true;
}

__global__ void gemm_kernel(const dtype *A, const dtype *B, dtype *C, int m, int k, int n)
{
    dtype c = 0.0;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n)
    {
        for (int i = 0; i < k; ++i)
        {
            c += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = c;
    }
}

int main(int argc, char **argv)
{
    // Set random alpha
    srand((unsigned)time(NULL));

    // Get args to Init matrix
    // Size of Matrix
    int m = atol(argv[1]); // Rows of Matrix A
    int k = atol(argv[2]); // Rows of Matrix B
    int n = atol(argv[3]); // Columns of Matrix B

    int nnz_A = m * k;
    int nnz_B = k * n;
    int nnz_C = m * n;

    cout << "marix A: " << m << " X " << k << "    marix B: " << k << " X " << n << endl;

    // seed
    int random_seed = atol(argv[4]);

    // Set random seed for generating matrix
    srand(random_seed);

    // Init Random Dense Matrix
    dtype *dev_matA;
    dtype *dev_matB;
    dtype *dev_matC;

    dtype *matA = (dtype *)malloc(nnz_A * sizeof(dtype));
    dtype *matB = (dtype *)malloc(nnz_B * sizeof(dtype));
    memset(matA, 0, nnz_A * sizeof(dtype));
    memset(matB, 0, nnz_B * sizeof(dtype));

    // Generate Dense Matrix
    generate_matrix(matA, nnz_A);
    generate_matrix(matB, nnz_B);

    cudaMalloc((void **)&dev_matA, nnz_A * sizeof(dtype));
    cudaMalloc((void **)&dev_matB, nnz_B * sizeof(dtype));
    cudaMalloc((void **)&dev_matC, nnz_C * sizeof(dtype));
    cudaMemcpy(dev_matA, matA, nnz_A * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matB, matB, nnz_B * sizeof(dtype), cudaMemcpyHostToDevice);

    // Set Device Id for the following Calculation
    cudaSetDevice(0);

    // Create timer_kernel
    my_timer timer_kernel;

    double time_use_kernel = 0.0;

    dim3 dimBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    dim3 dimGrid((n / dimBlock.x) + (n % dimBlock.x != 0), (m / dimBlock.y) + (m % dimBlock.y != 0));

    // Start timer_kernel
    timer_kernel.start();

    // Execute Device gemm
    gemm_kernel<<<dimGrid, dimBlock>>>(dev_matA, dev_matB, dev_matC, m, k, n);
    cudaDeviceSynchronize();

    // Stop timer kernel
    timer_kernel.stop();

    time_use_kernel += timer_kernel.time_use;
    dtype *matC = (dtype *)malloc(nnz_C * sizeof(dtype));
    cudaMemcpy(matC, dev_matC, nnz_C * sizeof(dtype), cudaMemcpyDeviceToHost);

    cudaFree(dev_matC);
    cudaFree(dev_matB);
    cudaFree(dev_matA);

    cout << "Device calculation finished!  elapsed time:" << time_use_kernel << "(us)" << endl;
    cout << "==================================================================" << endl;

    if (verify(matA, matB, matC, m, k, n))
    {
        cout << "PASS!" << endl;
    }
    else
    {
        cout << endl
             << "NOT PASS, CHECK YOUR CODE!" << endl;
    }
    free(matA);
    free(matB);
    free(matC);
    return 0;
}
