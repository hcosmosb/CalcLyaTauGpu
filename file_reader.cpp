#include "file_reader.hpp"
#include "constants.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <omp.h> 
#include <fstream>
#include <math.h>

std::vector<float> readBinaryFileHalf(const std::string& filename, int& nx, int& ny, int& nz, bool firstHalf) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: unable to open file" << filename << std::endl;
        exit(1);
    }

    int nbyte;

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(int));
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx)); nx/=2;
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    file.read(reinterpret_cast<char*>(&nz), sizeof(nz));
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));
    std::vector<float> data(nbyte / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), nbyte);
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));

    file.close();

    size_t half_size = nbyte / sizeof(float) / 2;
    if (firstHalf) {
        return std::vector<float>(data.begin(), data.begin() + half_size);
    } else {
        return std::vector<float>(data.begin() + half_size, data.end());
    }
    
    return data;
}

std::vector<float> readBinaryFile(const std::string& filename, int& nx, int& ny, int& nz) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: unable to open file" << filename << std::endl; exit(1);
    }

    int nbyte;

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(int));
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    file.read(reinterpret_cast<char*>(&nz), sizeof(nz));
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));
    std::vector<float> data(nbyte / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), nbyte);
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));
    
    file.close();

    return data;
}


// Function to read rho files
std::vector<std::vector<std::vector<float>>> readChunk(const std::string& dataType, int ichunk, bool firstHalf) {
    std::stringstream ss;
    if (isCoDaIII){
    ss << SIMDIR << "/"<<dataType<<"_" <<std::setw(5)<<std::setfill('0')<<ichunk;
    } else {
        ss << SIMDIR << "/"<<dataType<<ichunk;
    }
    std::string filename = ss.str();

    if (ichunk<std::pow(ndim/nchdim,2)){
        std::cout << "# Input file: " << filename << std::endl;
    } else {
        std::cout << "# " << std::setfill('0') << ichunk <<std::endl;
    }
    
    int nx, ny, nz;
    // std::vector<float> arr = readBinaryFile(filename, nx, ny, nz);
    std::vector<float> arr;
    if (isCoDaIII) {
        arr = readBinaryFileHalf(filename, nx, ny, nz, firstHalf);
    } else {
        arr = readBinaryFile(filename, nx, ny, nz);
    }
    // std::cout<<"!# nx, ny, nz:"<<nx<<", "<<ny<<", "<<nz<<std::endl;
    // std::cout<<"!# arr[-1] = "<<arr[nx*ny*nz-1]<<std::endl;
    return reshape(arr, nx, ny, nz);
}

// Function to concatenate chunks for a given type of data
std::vector<std::vector<std::vector<float>>> loadStick(const int ixch, const std::string& dataType, bool firstHalf) {
    std::vector<std::vector<std::vector<float>>> fullData;
    int nx, ny;
    // Read the first chunk to determine dimensions
    //std::vector<std::vector<std::vector<float>>> firstChunk;
    // firstChunk = readChunk(dataType,0);
    // int nx = firstChunk.size();
    // int ny = firstChunk[0].size();
    // int nx = 512, ny = 512;
    if (isCoDaIII){
        nx = 256, ny = 512;
    } else {
        nx = 512, ny = 512;
    }

    // Initialize fullData with empty z-dimension
    fullData.resize(nx, std::vector<std::vector<float>>(ny));

    // Loop through all chunks and concatenate
    // for (int i = ixch; i < std::pow(ndim/nchdim,2)-1; i += std::pow(ndim/nchdim,2)) {
    for (int i = ixch; i < nchunk; i += std::pow(ndim/nchdim,2)) {
        std::vector<std::vector<std::vector<float>>> chunk;
        chunk = readChunk(dataType, i, firstHalf); 
        
        // Append chunk data along the z direction
        #pragma omp parallel for
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy) 
                fullData[ix][iy].insert(fullData[ix][iy].end(), chunk[ix][iy].begin(), chunk[ix][iy].end());
    }
    return fullData;
}

void processRTV(std::vector<std::vector<std::vector<float>>>& nHI, 
                std::vector<std::vector<std::vector<float>>>& xion, 
                std::vector<std::vector<std::vector<float>>>& temp, 
                std::vector<std::vector<std::vector<float>>>& vz, 
                const int nx, const int ny, const int nz) {
    std::cout<<"# Processing the arrays ... "<<std::endl;
    std::vector<double> nHISumX(nx),tempSumX(nx),xionSumX(nx);
    #pragma omp parallel for
    for (int ix=0; ix<nx; ix++)
        for (int iy=0; iy<ny; iy++)
            for (int iz=0; iz<nz; iz++){
                temp[ix][iy][iz] *= tconv/(nHI[ix][iy][iz]*(1.+xion[ix][iy][iz])); //converting to K
                temp[ix][iy][iz] = std::max(temp[ix][iy][iz],10.0f);
                vz[ix][iy][iz]   *= vconv; // converting to km/s
                nHI[ix][iy][iz]  *= 8.*nHmean*(1.-xion[ix][iy][iz]); // converting to the unit of cosmic mean
                tempSumX[ix] += temp[ix][iy][iz];
                nHISumX[ix]  += nHI[ix][iy][iz];
            }

    double nHISum=0., tempSum=0.;
    for (int ix=0; ix<nx; ix++){
        nHISum  += nHISumX[ix];
        tempSum += tempSumX[ix];
    }
    std::cout<<"# nHI mean: "<<nHISum/nx/ny/nz<<std::endl;
    std::cout<<"# Temp mean: "<<tempSum/nx/ny/nz<<std::endl;
}

std::vector<std::vector<std::vector<float>>> reshape(const std::vector<float>& data, int nx, int ny, int nz) {
    std::vector<std::vector<std::vector<float>>> reshaped(nx, std::vector<std::vector<float>>(ny, std::vector<float>(nz)));
    float dummy;

    #pragma omp parallel for
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                // reshaped[i][j][k] = data[k+nz*j+nz*ny*i];
                reshaped[i][j][k] = data[i + nx*j + nx*ny*k];
    
    return reshaped;
}





void printRamUsage() {
    std::ifstream status_file("/proc/self/status");
    std::string line;

    while (std::getline(status_file, line)) {
        if (line.find("VmRSS:") == 0) {  // Resident Set Size (RAM usage)
            size_t pos = line.find(":");
            std::string valueStr = line.substr(pos + 1);
            // Remove "kB" and convert to integer
            valueStr.erase(valueStr.find("kB"), 2);
            int valueKB = std::atoi(valueStr.c_str());
            double valueGB = static_cast<double>(valueKB) / (1024 * 1024); // Convert KB to GB
            std::cout << "# RAM Usage: " << valueGB << " GB ";
        }
        if (line.find("VmSize:") == 0) {  // Virtual Memory Size
            size_t pos = line.find(":");
            std::string valueStr = line.substr(pos + 1);
            // Remove "kB" and convert to integer
            valueStr.erase(valueStr.find("kB"), 2);
            int valueKB = std::atoi(valueStr.c_str());
            double valueGB = static_cast<double>(valueKB) / (1024 * 1024); // Convert KB to GB
            std::cout << "# Virtual Memory Size: " << valueGB << " GB ";
        }
    }
    std::cout<<std::endl;
}


