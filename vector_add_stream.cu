#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>

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

    const int nStreams = 4;
    std::vector<cudaStream_t> streams(nStreams);
    for (int i = 0; i < nStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    int chunkSize = (N + nStreams - 1) / nStreams;

    int blockSize = 256;
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, 0));
    int gridSize = props.multiProcessorCount * 5;

    std::cout << "Starting pipelined execution with "
              << nStreams
              << " streams..."
              << std::endl;

    // Time the entire pipelined operation.
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * chunkSize;
        int currentSize = (i == nStreams - 1) ? (N - offset) : chunkSize;
        if (currentSize <= 0) continue;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, A + offset, currentSize * sizeof(float),
            cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b + offset, B + offset, currentSize * sizeof(float),
            cudaMemcpyHostToDevice, streams[i]));

        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset, currentSize);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(C + offset, d_c + offset, currentSize * sizeof(float),
            cudaMemcpyDeviceToHost, streams[i]));
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_ms, start, stop));
    std::cout << "Total Time for Pipelined Execution: "
              << std::fixed << std::setprecision(3) << total_ms << " ms" << std::endl;

    for (int i = 0; i < nStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    std::cout << "Verifying results..." << std::endl;
    bool success = true;
    for (int i = 0; i < N; i += N / 10) { // Check first 10.
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
