#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define WIDTH 800
#define HEIGHT 600

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

__global__ void PictureKernel(const unsigned char *input, unsigned char *output,
                              int height, int width)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        const int index = row * width + col;
        const int scaled = 2 * input[index];
        output[index] = static_cast<unsigned char>(scaled > 255 ? 255 : scaled);
    }
}

int main()
{
    const size_t count = static_cast<size_t>(WIDTH) * HEIGHT;
    const size_t bytes = count * sizeof(unsigned char);
    auto *input = static_cast<unsigned char *>(std::malloc(bytes));
    auto *output = static_cast<unsigned char *>(std::malloc(bytes));
    unsigned char *d_input = nullptr, *d_output = nullptr;
    if (!input || !output) {
        std::fprintf(stderr, "Could not allocate host pixel arrays.\n");
        std::free(input); std::free(output);
        return 1;
    }
    for (size_t i = 0; i < count; ++i)
        input[i] = static_cast<unsigned char>(std::rand() % 256);

    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, bytes));
    CHECK(cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice));

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    std::printf("Grid: %u x %u blocks of %u x %u threads\n",
                numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    const double started = cpuSecond();
    PictureKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, HEIGHT, WIDTH);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    const double elapsed = cpuSecond() - started;
    CHECK(cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost));

    std::printf("Original top-left 50 x 50:\n");
    for (int row = 0; row < 50; ++row) {
        for (int col = 0; col < 50; ++col)
            std::printf("%3u ", static_cast<unsigned int>(input[row * WIDTH + col]));
        std::printf("\n");
    }
    std::printf("Updated top-left 50 x 50:\n");
    for (int row = 0; row < 50; ++row) {
        for (int col = 0; col < 50; ++col)
            std::printf("%3u ", static_cast<unsigned int>(output[row * WIDTH + col]));
        std::printf("\n");
    }

    size_t capped = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        const int scaled = 2 * static_cast<int>(input[i]);
        const unsigned char expected = static_cast<unsigned char>(scaled > 255 ? 255 : scaled);
        if (scaled > 255)
            ++capped;
        if (output[i] != expected)
            ++mismatches;
    }
    std::printf("Verification: %zu mismatches out of %zu\n", mismatches, count);
    std::printf("Pixels capped at 255: %zu\n", capped);
    std::printf("Time taken - %.7f\n", elapsed);

    std::free(input); std::free(output);
    CHECK(cudaFree(d_input)); CHECK(cudaFree(d_output));
    return mismatches == 0 ? 0 : 1;
}
