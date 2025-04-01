#include <iostream>
#include <cuda_runtime.h>

void checkGPUs(){
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus == 0){
        std::cerr << "# No GPUs found!" << std::endl;
        return;
    }
    std::cout << "# Number of available GPUs: " << num_gpus << std::endl;
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        std::cout << "# GPU " << i << ": " << prop.name << std::endl;
        std::cout << "# Multiprocessor Count: " << prop.multiProcessorCount <<std::endl;
        std::cout << "# Maximum number of threads per processor: " << prop.maxThreadsPerMultiProcessor <<std::endl;
    }
    return;
}
