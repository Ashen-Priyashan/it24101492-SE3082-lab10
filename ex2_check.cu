#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

#define CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::printf("Error: %s:%d, code:%d, reason: %s\n", \
                    __FILE__, __LINE__, static_cast<int>(error), cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

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
    int *a = nullptr, *b = nullptr, *c = nullptr;
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    const size_t bytes = static_cast<size_t>(N) * sizeof(int);
    double started, elapsed;

#ifdef EXPERIMENT_BAD_ALLOC
    CHECK(cudaMalloc(&d_a, static_cast<size_t>(1) << 50));
#else
    CHECK(cudaMalloc(&d_a, bytes));
#endif
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    a = static_cast<int *>(std::malloc(bytes));
    b = static_cast<int *>(std::malloc(bytes));
    c = static_cast<int *>(std::malloc(bytes));
    if (!a || !b || !c) {
        std::fprintf(stderr, "Could not allocate host arrays.\n");
        return 1;
    }
    random_ints(a, N);
    random_ints(b, N);
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    started = cpuSecond();
#ifdef EXPERIMENT_BAD_LAUNCH
    add<<<N, 2048>>>(d_a, d_b, d_c);
#else
    add<<<N, 1>>>(d_a, d_b, d_c);
#endif
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    elapsed = cpuSecond() - started;
    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; ++i)
        std::printf("%d + %d = %d\n", a[i], b[i], c[i]);
    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (c[i] != a[i] + b[i])
            ++errors;
    std::printf("Verification: %d mismatches out of %d\n", errors, N);
    std::printf("Time taken - %.7f\n", elapsed);

    std::free(a); std::free(b); std::free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return errors == 0 ? 0 : 1;
}
