#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cstddef>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void vecMul(const int *a, const int *b, int *c, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        c[i] = a[i] * b[i];
}

static bool check_cuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

static void random_ints(int *values, int count)
{
    for (int i = 0; i < count; ++i)
        values[i] = std::rand() % 100;
}

static double cpuSecond()
{
    timeval time_value{};
    gettimeofday(&time_value, nullptr);
    return static_cast<double>(time_value.tv_sec) + time_value.tv_usec * 1.0e-6;
}

int main()
{
    double dStart = 0.0, dElaps = 0.0;
    const size_t bytes = static_cast<size_t>(N) * sizeof(int);
    int *a = static_cast<int *>(std::malloc(bytes));
    int *b = static_cast<int *>(std::malloc(bytes));
    int *c = static_cast<int *>(std::malloc(bytes));
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (!a || !b || !c) {
        std::fprintf(stderr, "Could not allocate the host vectors.\n");
        std::free(a); std::free(b); std::free(c);
        return 1;
    }
    random_ints(a, N);
    random_ints(b, N);

    if (!check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)") ||
        !check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)") ||
        !check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)"))
        return 1;
    if (!check_cuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "copy a to device") ||
        !check_cuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "copy b to device"))
        return 1;

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    std::printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);
    dStart = cpuSecond();
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    if (!check_cuda(cudaGetLastError(), "vecMul kernel launch")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "vecMul kernel execution")) return 1;
    dElaps = cpuSecond() - dStart;
    if (!check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "copy result to host")) return 1;

    for (int i = N - 1000; i < N; ++i)
        std::printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (c[i] != a[i] * b[i])
            ++errors;
    std::printf("Verification: %d mismatches out of %d\n", errors, N);
    std::printf("Time taken - %.7f\n", dElaps);

    std::free(a); std::free(b); std::free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors == 0 ? 0 : 1;
}
