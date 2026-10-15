#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <sys/time.h>

#define NX 500
#define NY 500
#define NZ 500

#define CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::printf("Error: %s:%d, code:%d, reason: %s\n", \
                    __FILE__, __LINE__, static_cast<int>(error), cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

static double cpuSecond()
{
    timeval time_value{};
    gettimeofday(&time_value, nullptr);
    return static_cast<double>(time_value.tv_sec) + time_value.tv_usec * 1.0e-6;
}

__global__ void mulElem3D(const int *a, const int *b, int *c,
                          int nx, int ny, int nz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = (static_cast<size_t>(z) * ny + y) * nx + x;
        c[index] = a[index] * b[index];
    }
}

static void random_ints(int *values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        values[i] = std::rand() % 100;
}

int main()
{
    const size_t count = static_cast<size_t>(NX) * NY * NZ;
    const size_t bytes = count * sizeof(int);
    int *a = static_cast<int *>(std::malloc(bytes));
    int *b = static_cast<int *>(std::malloc(bytes));
    int *c = static_cast<int *>(std::malloc(bytes));
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (!a || !b || !c) {
        std::fprintf(stderr, "Could not allocate the host volumes.\n");
        std::free(a); std::free(b); std::free(c);
        return 1;
    }
    random_ints(a, count);
    random_ints(b, count);

    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    const dim3 threadsPerBlock(8, 8, 8);
    const dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                         (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);
    std::printf("Grid: %u x %u x %u blocks\n", numBlocks.x, numBlocks.y, numBlocks.z);

    const double started = cpuSecond();
    mulElem3D<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, NX, NY, NZ);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    const double elapsed = cpuSecond() - started;
    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (size_t index = count - 1000; index < count; ++index) {
        const size_t x = index % NX;
        const size_t y = (index / NX) % NY;
        const size_t z = index / (static_cast<size_t>(NX) * NY);
        std::printf("C[%zu][%zu][%zu] = %d x %d = %d\n",
                    z, y, x, a[index], b[index], c[index]);
    }

    size_t errors = 0;
    for (size_t i = 0; i < count; ++i)
        if (c[i] != a[i] * b[i])
            ++errors;
    std::printf("Verification: %zu mismatches out of %zu\n", errors, count);
    std::printf("Time taken - %.7f\n", elapsed);

    std::free(a); std::free(b); std::free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return errors == 0 ? 0 : 1;
}
