#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cstddef>

#define ROWS 10000
#define COLS 10000

__global__ void mulElem(const int *a, const int *b, int *c, int rows, int cols)
{
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < rows && col < cols) {
        const int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

static bool check_cuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

static void random_ints(int *values, size_t count)
{
    for (size_t i = 0; i < count; ++i)
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
    const size_t count = static_cast<size_t>(ROWS) * COLS;
    const size_t bytes = count * sizeof(int);
    int *a = static_cast<int *>(std::malloc(bytes));
    int *b = static_cast<int *>(std::malloc(bytes));
    int *c = static_cast<int *>(std::malloc(bytes));
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (!a || !b || !c) {
        std::fprintf(stderr, "Could not allocate the host matrices.\n");
        std::free(a); std::free(b); std::free(c);
        return 1;
    }
    random_ints(a, count);
    random_ints(b, count);

    if (!check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)") ||
        !check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)") ||
        !check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)"))
        return 1;
    if (!check_cuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "copy a to device") ||
        !check_cuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "copy b to device"))
        return 1;

    const dim3 threadsPerBlock(32, 16);
    const dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);
    std::printf("Grid: %u x %u blocks\n", numBlocks.x, numBlocks.y);
    dStart = cpuSecond();
    mulElem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, ROWS, COLS);
    if (!check_cuda(cudaGetLastError(), "mulElem kernel launch")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "mulElem kernel execution")) return 1;
    dElaps = cpuSecond() - dStart;
    if (!check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "copy result to host")) return 1;

    const int last_row = ROWS - 1;
    for (int col = COLS - 1000; col < COLS; ++col) {
        const size_t idx = static_cast<size_t>(last_row) * COLS + col;
        std::printf("C[%d][%d] = %d x %d = %d\n", last_row, col, a[idx], b[idx], c[idx]);
    }

    int errors = 0;
    for (size_t i = 0; i < count; ++i)
        if (c[i] != a[i] * b[i])
            ++errors;
    std::printf("Verification: %d mismatches out of %zu\n", errors, count);
    std::printf("Time taken - %.7f\n", dElaps);

    std::free(a); std::free(b); std::free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return errors == 0 ? 0 : 1;
}
