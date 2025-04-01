// module load cudatoolkit
// nvcc -o testRead constants.cpp file_reader.cpp testRead.cu && ./testRead 
#include <iostream>
#include <chrono>
#include "constants.hpp"
#include "file_reader.hpp"
#include <cuda_runtime.h>

#define NX 1024
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
    //int size_input = NX * NY * NZ * sizeof(float);
    //int size_output = NY * NZ * sizeof(float);
    size_t size_input = static_cast<size_t>(NX) * NY * NZ * sizeof(float);
    size_t size_output = static_cast<size_t>(NY) * NZ * sizeof(float);

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
    std::cout << "# GPU computation time: " << elapsed.count() << " seconds\n";

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

void checkGPUs(){
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus == 0){
        std::cerr << "No GPUs found!" << std::endl;
        return;
    }
    std::cout << "Number of available GPUs: " << num_gpus << std::endl;
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
    }
    return;
}


int main() {

    checkGPUs();

    // auto xion = concatenateChunks(false);
    // std::cout << "# xion loaded with dimensions: "<<xion.size()<<" x "<<xion[0].size()<<" y "<<xion[0][0].size()<<" z "<<std::endl;
    auto rho = concatenateChunks(true);    
    std::cout << "# rho loaded with dimensions: " << rho.size() << " x " << rho[0].size() << " y " << rho[0][0].size() << " z " << std::endl;
    int nx = rho.size(); int ny = rho[0].size(); int nz = rho[0][0].size();

    std::vector<std::vector<float>> meanRho(ny, std::vector<float>(nz, 0.0f));

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
            for (int i = 0; i < nx; i++)
                meanRho[j][k] += rho[i][j][k] / nx;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "# Computation time on CPU is " << elapsed.count() << " seconds." << std::endl;
    std::cout << "# mean at (iy, iz) = (0, 0): " << meanRho[0][0] << std::endl;
    std::cout << "# mean at (iy, iz) = (0, 1): " << meanRho[0][1] << std::endl;
    std::cout << "# mean at (iy, iz) = (0, 2): " << meanRho[0][2] << std::endl;
    std::cout << "# mean at (iy, iz) = (0, 3): " << meanRho[0][3] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    computeMeanGPU(rho, meanRho);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "# Computation time on GPU is " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
