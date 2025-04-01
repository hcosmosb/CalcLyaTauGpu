// module load cudatoolkit
// nvcc -o main sig_a_app.cpp checkGPUs.cpp constants.cpp file_reader.cpp calcTrGPU.cu main.cu && ./main
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "constants.hpp"
#include "file_reader.hpp"
#include "checkGPUs.hpp"
double sig_a_app(double dl_a_Angst,double T_in);
void calcTrGPU(const std::vector<float>& rhoChunk, const std::vector<float>& xionChunk, std::vector<float>& tauChunk, int N);
void calcTrGPU2D(const std::vector<std::vector<float>>& nHI, const std::vector<std::vector<float>>& temp, const std::vector<std::vector<float>>& vz, std::vector<std::vector<float>>& tau, int NY, int NZ);

void calcTrCPU(std::vector<float>& nHIChunk, std::vector<float>& tempChunk, std::vector<float>& vzChunk, std::vector<float>& tauChunk, int N){
    double dl, sig;
    int dcell;
    
    for (int irho=0; irho<N; irho++){
        for (int itau=0; itau<N; itau++){
            dcell = (itau - irho + 3*N/2)%N - N/2;
            dl = dcell*dvaCell/(c_sl/1e5)*lLya + dlPerKmps*vzChunk[irho];
            sig = sig_a_app(dl, tempChunk[irho]);
            tauChunk[itau] += nHIChunk[irho]*cell_cm*sig;
        }
    }
}

void printLOS(std::vector<float> tauLOS, int N){ //Print tau values at several cells for debugging.
    std::cout<<"# tauLOS[0]:    "<<tauLOS[0]<<std::endl; 
    std::cout<<"# tauLOS[1]:    "<<tauLOS[1]<<std::endl; 
    std::cout<<"# tauLOS[10]:   "<<tauLOS[10]<<std::endl; 
    std::cout<<"# tauLOS[100]:  "<<tauLOS[100]<<std::endl; 
    std::cout<<"# tauLOS[511]:  "<<tauLOS[511]<<std::endl;
}

void writeOutput(std::vector<std::vector<std::vector<float>>>& arr, int nx, int ny, int nz, const std::string& oFileName){
    std::ofstream outFile(oFileName,std::ios::binary);
    if (!outFile) {std::cerr<<"# Failed to open "<<oFileName<<" for writing."<<std::endl; return;}
    
    outFile.write(reinterpret_cast<char*>(&nx), sizeof(int));
    outFile.write(reinterpret_cast<char*>(&ny), sizeof(int));
    outFile.write(reinterpret_cast<char*>(&nz), sizeof(int));
    for (int ix=0; ix<nx; ix++)
        for (int iy=0; iy<ny; iy++)
            outFile.write(reinterpret_cast<char*>(arr[ix][iy].data()),nz*sizeof(float));
    outFile.close(); 
    std::cout<<"# File saved at "<<oFileName<<std::endl;
}

// Setup for CoDaIII simulation data at z = 5.9
void codaiiiSetup(){
    isCoDaIII = true;
    SIMDIR = "(input directory)";
    OUTDIR = "(input directory)";
    unit_l = 4.19205590504315e+25; unit_d = 8.85801687230084e-28; unit_t = 9.440132939404150E+15;
    isn = 82; ndim = 8192; nchdim = 512;
    Lbox = 64.;
    z = 5.9385316086438085;
}

// Setup for CoDaII simulation data at z = 6.5
void codaiiSetup(){
    isCoDaIII = false;
    SIMDIR = "(input directory)";
    OUTDIR = "(input directory)";
    unit_l = 3.85025454589881E+25; unit_d = 1.14325652730409E-27; unit_t = 7.96354892608204E+15;
    isn = 78; ndim = 4096; nchdim = 512;
    Lbox = 64.;
    z = 6.55445814;
}

int main(int argc, char* argv[]) {
    int ixch = 1; bool firstHalf = true;
    // Parse command-line arguments
    if (argc > 1) ixch = std::atoi(argv[1]);  // Convert first argument to integer
    // Due to the memory limit of the Perlmutter cluster, we have process only half of the data at a time.
    if (argc > 2) firstHalf = std::atoi(argv[2]);  // Convert second argument to boolean (0 or 1)
    // codaiiiSetup();
    codaiiSetup();
    checkGPUs(); printConstants();


    auto start = std::chrono::high_resolution_clock::now();
    auto nHI = loadStick(ixch, "rho", firstHalf); int nx = nHI.size(); int ny = nHI[0].size(); int nz = nHI[0][0].size(); printRamUsage();
    std::cout << "# Arrays dimensions: (" << nx << ", " << ny << ", " << nz << ") " << std::endl;
    auto xion = loadStick(ixch, "xion", firstHalf); printRamUsage();
    auto temp = loadStick(ixch, "temp", firstHalf); printRamUsage();
    auto vz   = loadStick(ixch, "vz",   firstHalf); printRamUsage();
    processRTV(nHI, xion, temp, vz, nx, ny, nz);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "# Time for reading and processing the data: " << elapsed.count() << " seconds\n"; // 162.024 seconds

    std::vector<float> tauLOS(nz, 0.0f);
    start = std::chrono::high_resolution_clock::now();
    for (int ix=128; ix<129; ix++)
        calcTrCPU(nHI[ix][0],temp[ix][0],vz[ix][0], tauLOS,nz);

    end = std::chrono::high_resolution_clock::now(); elapsed = end - start; elapsed = end - start;
    std::cout << "# CPU computation time for one LOS: " << elapsed.count() << " seconds\n"; //6.1797 seconds
    printLOS(tauLOS,nz);


    std::vector<std::vector<std::vector<float>>> tau(nx, std::vector<std::vector<float>>(ny, std::vector<float>(nz, 0.0f)));
    start = std::chrono::high_resolution_clock::now();
    for (int ix=0; ix<nx; ix++){
    // for (int ix=128; ix<129; ix++){
        if (ix%32==0) {
            end = std::chrono::high_resolution_clock::now(); elapsed = end - start;
            std::cout<<"# working on ix = "<<ix<<"; t = "<< elapsed.count()<<" secs."<<std::endl;
        }
        calcTrGPU2D(nHI[ix], temp[ix], vz[ix], tau[ix], ny, nz); //509.25 seconds. slightly faster.
    }
    end = std::chrono::high_resolution_clock::now(); elapsed = end - start;
    std::cout << "# GPU computation time: " << elapsed.count() << " seconds\n";

    // printLOS(tau[0][0],nz); printLOS(tau[0][256],nz); 
    printLOS(tau[128][0],nz);

    std::ostringstream oss;
    if (isCoDaIII){
        oss << OUTDIR << "/tau_z_" << std::setw(3) << std::setfill('0') << ixch << (firstHalf ? "u" : "d") << ".bin";
    } else {
        oss << OUTDIR << "/tau_z_" << std::setw(2) << std::setfill('0') << ixch << ".bin";
    }
    
    std::string outFile = oss.str(); // Convert to std::string
    writeOutput(tau, nx, ny, nz, outFile);
    return 0;
}


// # tauLOS[0]:    23.1374
// # tauLOS[1]:    23.168
// # tauLOS[10]:   23.2049
// # tauLOS[100]:  1183.06
// # tauLOS[511]:  49.0593

// # tauLOS[0]:    9.22867
// # tauLOS[1]:    8.46004
// # tauLOS[10]:   5.80764
// # tauLOS[100]:  22.8562
// # tauLOS[511]:  62.5693
