#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void add(const int *a, const int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
        std::fprintf(stderr, "Could not allocate host arrays.\n");
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

    dStart = cpuSecond();
    add<<<N, 1>>>(d_a, d_b, d_c);
    if (!check_cuda(cudaGetLastError(), "add kernel launch")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "add kernel execution")) return 1;
    dElaps = cpuSecond() - dStart;
    if (!check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "copy result to host")) return 1;

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        std::printf("%d + %d = %d\n", a[i], b[i], c[i]);
        if (c[i] != a[i] + b[i])
            ++errors;
    }
    std::printf("Verification: %d mismatches out of %d\n", errors, N);
    std::printf("Time taken - %.7f\n", dElaps);

    std::free(a); std::free(b); std::free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors == 0 ? 0 : 1;
}
