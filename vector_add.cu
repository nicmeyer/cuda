#include <cassert>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = startIndex; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 24;

    float *A, *B, *C;

    CHECK_CUDA_ERROR(cudaMallocHost(&A, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&B, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C, N * sizeof(float)));

    for (int i = 0; i < N; ++i) {
        A[i] = float(i);
        B[i] = float(i * 2);
    }

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));

    // Time the Host-to-Device memory copy.
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float memcpyHtoD_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&memcpyHtoD_ms, start, stop));
    std::cout << "Time for Host->Device Copy: "
              << std::fixed
              << std::setprecision(3)
              << memcpyHtoD_ms
              << " ms"
              << std::endl;

    int blockSize = 256;
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, 0));
    std::cout << "Number of Streaming Multiprocessors (SMs): "
              << props.multiProcessorCount << std::endl;
    int gridSize = props.multiProcessorCount * 5; // (N + blockSize - 1) / blockSize;

    // Time the Kernel Execution.
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_ms, start, stop));
    std::cout << "Time for Kernel Execution: "
              << std::fixed
              << std::setprecision(3)
              << kernel_ms
              << " ms"
              << std::endl;

    // Time the Device-to-Host memory copy.
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float memcpyDtoH_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&memcpyDtoH_ms, start, stop));
    std::cout << "Time for Device->Host Copy: "
              << std::fixed
              << std::setprecision(3)
              << memcpyDtoH_ms
              << " ms"
              << std::endl;

    std::cout << "Verifying results..." << std::endl;
    bool success = true;
    for (int i = 0; i < 10; ++i) { // Check first 10.
        if (std::abs(C[i] - (A[i] + B[i])) > 1e-5) {
            std::cerr << "Verification failed at index " << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        for (int i = N - 10; i < N; ++i) { // Check last 10.
            if (std::abs(C[i] - (A[i] + B[i])) > 1e-5) {
                std::cerr << "Verification failed at index " << i << std::endl;
                success = false;
                break;
            }
        }
    }

    if (success) {
        std::cout << "Verification successful!" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFreeHost(A));
    CHECK_CUDA_ERROR(cudaFreeHost(B));
    CHECK_CUDA_ERROR(cudaFreeHost(C));

    return 0;
}
