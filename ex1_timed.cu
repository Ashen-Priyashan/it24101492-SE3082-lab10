#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void add(const int *a, const int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

static double cpuSecond()
{
    timeval time_value{};
    gettimeofday(&time_value, nullptr);
    return static_cast<double>(time_value.tv_sec) + time_value.tv_usec * 1.0e-6;
}

static void random_ints(int *values, int count)
{
    for (int i = 0; i < count; ++i)
        values[i] = std::rand() % 100;
}

int main()
{
    const size_t bytes = static_cast<size_t>(N) * sizeof(int);
    int *a = static_cast<int *>(std::malloc(bytes));
    int *b = static_cast<int *>(std::malloc(bytes));
    int *c = static_cast<int *>(std::malloc(bytes));
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (!a || !b || !c) {
        std::fprintf(stderr, "Could not allocate host arrays.\n");
        return 1;
    }
    random_ints(a, N);
    random_ints(b, N);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    const double started = cpuSecond();
#ifdef EXPERIMENT_BAD_LAUNCH
    add<<<N, 2048>>>(d_a, d_b, d_c);
#else
    add<<<N, 1>>>(d_a, d_b, d_c);
#endif
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
        std::printf("Kernel launch failed: %s\n", cudaGetErrorString(launch_error));
#ifndef OMIT_DEVICE_SYNC
    cudaDeviceSynchronize();
#endif
    const double elapsed = cpuSecond() - started;

    // This copy waits for the GPU even in the no-synchronize experiment.
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; ++i)
        std::printf("%d + %d = %d\n", a[i], b[i], c[i]);
    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (c[i] != a[i] + b[i])
            ++errors;
    std::printf("Verification: %d mismatches out of %d\n", errors, N);
    std::printf("Time taken - %.7f\n", elapsed);

    std::free(a); std::free(b); std::free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
