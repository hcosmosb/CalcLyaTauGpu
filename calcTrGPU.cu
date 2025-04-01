#include <iostream>
#include <vector>
#include "constants.hpp"
#include <omp.h> 

#define NGY 128

__device__ __forceinline__ double sig_a_app(const double dl_a_Angst, const double T_in) {
    const double c_sl = 2.99792e10;  // speed of light in cm/s
    const double k_boltz = 1.380648e-16;  // Boltzmann constant in erg/K
    const double m_p = 1.6726219e-24;  // proton mass in g
    const double l_a_Angst = 1215.67;  // Lyman-alpha wavelength in Angstroms

    // Calculations
    double T4emh = pow(T_in / 1e4, -0.5);
    double a_V = 4.7e-4 * T4emh;
    double dnu_a = c_sl / ((dl_a_Angst + l_a_Angst) * 1e-8) - c_sl / (l_a_Angst * 1e-8);
    double Delnu_D = 2.46e15 * sqrt(2.0 * k_boltz * T_in / (m_p * c_sl * c_sl));

    double x = fabs(dnu_a / Delnu_D);
    double x2 = x * x;
    double z = (x2 - 0.855) / (x2 + 3.42);

    double q;
    if (z <= 0) {
        q = 0.0;
    } else {
        q = z * (1 + 21 / x2) * a_V / M_PI / (x2 + 1) * (0.1117 + z * (4.421 + z * (-9.207 + 5.674 * z)));
    }

    double phi_app = q + exp(-x2) / sqrt(M_PI);

    return phi_app * 5.889e-14 * T4emh;
}

__global__ void computeTau2D(const float* nHI, const float* temp, const float* vz, float* output, int NY, int NZ, double dvaCell, double cell_cm, double dlPerKmps) {
    int irho = blockIdx.y * blockDim.x + threadIdx.x;
    int iy = blockIdx.x;
    double sig, dl, delta_tau;
    int dcell, itau, rInd;
    double lLya=1215.67, c_sl_kmps=2.99792e5;
    double dlfac = dvaCell/(c_sl_kmps)*lLya;

    if (irho < NZ){
        for (int itau2=0; itau2<NZ; itau2++){
            itau = (itau2+irho)%NZ;
            dcell = (itau-irho+3*NZ/2)%NZ - NZ/2;

            rInd = NZ*iy + irho;
            dl = dcell*dlfac + dlPerKmps*vz[rInd];
            sig = sig_a_app(dl,temp[rInd]);
            delta_tau = nHI[rInd] * cell_cm * sig;
            //if(itau==100 && iy==0 && abs(dl)<0.3)printf("!!! dcell = %i; delta_tau = %f; sig = %f; dl = %f, temp = %f \n",dcell,delta_tau,sig*1e20,dl,temp[rInd]);
            atomicAdd(&output[NZ*iy + itau], delta_tau);
        }        
    }
}

void calcTrGPU2D(const std::vector<std::vector<float>>& nHI, 
                 const std::vector<std::vector<float>>& temp, 
                 const std::vector<std::vector<float>>& vz, 
                 std::vector<std::vector<float>>& tau, int NY, int NZ){
    int nGPU;
    cudaGetDeviceCount(&nGPU);
    std::vector<cudaStream_t> streams(nGPU);

    std::vector<float*> d_input_nHI(nGPU), d_input_temp(nGPU), d_input_vz(nGPU), d_output(nGPU);
    int chunkSize = NY * NZ / nGPU; // Divide data among GPUs

    size_t size_input  = static_cast<size_t>(chunkSize) * sizeof(float);
    size_t size_output = static_cast<size_t>(chunkSize) * sizeof(float);

    // Data flattening
    std::vector<float> nHIFlat(NY * NZ), tempFlat(NY * NZ), vzFlat(NY * NZ);
    #pragma omp parallel for
    for (int iy = 0; iy < NY; iy++)
        for (int iz = 0; iz < NZ; iz++){
            nHIFlat [iy*NZ + iz] = nHI [iy][iz];
            tempFlat[iy*NZ + iz] = temp[iy][iz];    
            vzFlat  [iy*NZ + iz] = vz  [iy][iz];
        }

    for (int igpu = 0; igpu < nGPU; igpu++) {
        cudaSetDevice(igpu);
        cudaStreamCreate(&streams[igpu]);
        // GPU memory allocation
        cudaMalloc(&d_input_nHI[igpu],  size_input);
        cudaMalloc(&d_input_temp[igpu], size_input);
        cudaMalloc(&d_input_vz[igpu],   size_input);
        cudaMalloc(&d_output[igpu],     size_output);
        cudaMemset(d_output[igpu],   0, size_output);

        // Copy chunks of data asynchronously
        cudaMemcpyAsync(d_input_nHI[igpu],  nHIFlat.data()  + igpu*chunkSize, size_input, cudaMemcpyHostToDevice, streams[igpu]);
        cudaMemcpyAsync(d_input_temp[igpu], tempFlat.data() + igpu*chunkSize, size_input, cudaMemcpyHostToDevice, streams[igpu]);
        cudaMemcpyAsync(d_input_vz[igpu],   vzFlat.data()   + igpu*chunkSize, size_input, cudaMemcpyHostToDevice, streams[igpu]);

        // Kernel configuration
        dim3 blockSize(1024);
        dim3 gridSize(NY/nGPU, (NZ+blockSize.x-1)/blockSize.x);

        // Kernel launch
        computeTau2D<<<gridSize, blockSize, 0, streams[igpu]>>>(d_input_nHI[igpu], d_input_temp[igpu], d_input_vz[igpu], d_output[igpu], NY, NZ, dvaCell, cell_cm, dlPerKmps);
    }
    // Copy device result to flattened data
    std::vector<float> tauFlat(NY*NZ);
    for (int igpu = 0; igpu < nGPU; igpu++) {
        cudaSetDevice(igpu);
        cudaMemcpyAsync(tauFlat.data() + igpu * chunkSize, d_output[igpu], chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[igpu]);
        cudaFree(d_input_nHI[igpu]);
        cudaFree(d_input_temp[igpu]);
        cudaFree(d_input_vz[igpu]);
        cudaFree(d_output[igpu]);
        cudaStreamDestroy(streams[igpu]);
    }
    //std::cout<<"!!! tauFlat[0-2]"<<tauFlat[0]<<" "<<tauFlat[1]<<" "<<tauFlat[2]<<" "<<std::endl;
    // Data reshaping
    #pragma omp parallel for
    for (int iy=0; iy < NY; iy++)
        for (int iz=0; iz < NZ; iz++)
            tau[iy][iz] = tauFlat[iy*NZ + iz];

}

