#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define NX 2048
#define NY 512
#define NZ 512

// CUDA kernel to compute mean along x-axis
__global__ void compute_mean(const float* input, float* output, int nx, int ny, int nz) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < ny && z < nz) {
        float sum = 0.0f;
        for (int x = 0; x < nx; ++x) {
            sum += input[x * ny * nz + y * nz + z];
        }
        output[y * nz + z] = sum / nx;
    }
}

// Function to copy data to device, launch kernel, and retrieve results
void computeMeanGPU(const std::vector<std::vector<std::vector<float>>>& hostData, 
                    std::vector<std::vector<float>>& meanData) {
    float *d_input, *d_output;
    int size_input = NX * NY * NZ * sizeof(float);
    int size_output = NY * NZ * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_output, size_output);

    // Flatten host data
    std::vector<float> flatData(NX * NY * NZ);
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z)
                flatData[x * NY * NZ + y * NZ + z] = hostData[x][y][z];

    // Copy input data to device
    cudaMemcpy(d_input, flatData.data(), size_input, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((NY + blockSize.x - 1) / blockSize.x, (NZ + blockSize.y - 1) / blockSize.y);

    // Measure GPU execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    compute_mean<<<gridSize, blockSize>>>(d_input, d_output, NX, NY, NZ);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "GPU computation time: " << elapsed.count() << " seconds\n";

    // Copy result back to host
    std::vector<float> flatOutput(NY * NZ);
    cudaMemcpy(flatOutput.data(), d_output, size_output, cudaMemcpyDeviceToHost);

    // Reshape data
    for (int y = 0; y < NY; ++y)
        for (int z = 0; z < NZ; ++z)
            meanData[y][z] = flatOutput[y * NZ + z];

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
