//nvcc -o testRead testRead.cpp && ./testRead
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <chrono>

const std::string SIMDIR = std::string(std::getenv("SCRATCH")) + "/Pierre/CoDaIII/prod_sr/reduced/fullbox";
const int isn = 60;
const int ichunk_size = 512;
const int nchunk = 4096;
const int nchdim = 2;

std::vector<float> readBinaryFile(const std::string& filename, int& nx, int& ny, int& nz) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: unable to open file" << filename << std::endl;
        exit(1);
    }

    int nbyte;

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(int));
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    file.read(reinterpret_cast<char*>(&nz), sizeof(nz));
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));

    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));
    std::vector<float> data(nbyte/sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), nbyte);
    file.read(reinterpret_cast<char*>(&nbyte), sizeof(nbyte));

    file.close();

    return data;
}

std::vector<std::vector<std::vector<float>>> reshape(const std::vector<float>& data, int nx, int ny, int nz){
    std::vector<std::vector<std::vector<float>>> reshaped(nx, std::vector<std::vector<float>>(ny, std::vector<float>(nz)));

    int index=0;
    
    for (int i=0; i<nx; i++) 
        for (int j=0; j<ny; j++) 
            for (int k=0; k<nz; k++)
                reshaped[i][j][k] = data[index++];

    return reshaped;
}

std::vector<std::vector<std::vector<float>>> readRhoChunk(int ichunk) {
    std::stringstream ss;
    ss<<SIMDIR<<"/output_"<<std::setw(6)<<std::setfill('0')<<isn<<"/rho_"<<std::setw(5)<<std::setfill('0')<<ichunk;
    std::string filename=ss.str();

    std::cout<<"# Input file: "<<filename<<std::endl;
    int nx,ny,nz;
    std::vector<float> rho = readBinaryFile(filename,nx,ny,nz);
    
    for (auto& val : rho) val *= 8.0f;
    return reshape(rho,nx,ny,nz);
}

std::vector<std::vector<std::vector<float>>> readXionChunk(int ichunk){
    std::stringstream ss;
    ss<<SIMDIR<<"/output_"<<std::setw(6)<<std::setfill('0')<<isn<<"/xion_"<<std::setw(5)<<std::setfill('0')<<ichunk;
    std::string filename=ss.str();

    std::cout<<"# Input file: "<<filename<<std::endl;
    int nx,ny,nz;
    std::vector<float> xion = readBinaryFile(filename,nx,ny,nz);
    return reshape(xion,nx,ny,nz);
}

std::vector<std::vector<std::vector<float>>> concatenateChunks(bool isRho){
    std::vector<std::vector<std::vector<float>>> fullData;

    for (int i=0; i<nchdim; i++){
        std::vector<std::vector<std::vector<float>>> chunk = isRho ? readRhoChunk(i) : readXionChunk(i);
        fullData.insert(fullData.end(),chunk.begin(),chunk.end());
    }
    return fullData;
}


int main(){
    // auto xion = concatenateChunks(false);
    // std::cout << "# xion loaded with dimensions: "<<xion.size()<<" x "<<xion[0].size()<<" y "<<xion[0][0].size()<<" z "<<std::endl;
    auto rho = concatenateChunks(true);
    std::cout << "# rho loaded with dimensions: "<<rho.size()<<" x "<<rho[0].size()<<" y "<<rho[0][0].size()<<" z "<<std::endl;
    
    int nx=rho.size(); int ny=rho[0].size(); int nz=rho[0][0].size();

    std::vector<std::vector<float>> meanRho(ny,std::vector<float>(nz,0.0f));

    auto start = std::chrono::high_resolution_clock::now();

    for (int j=0;j<ny;j++)
        for (int k=0;k<nz;k++)
            for (int i=0;i<nx;i++)
                meanRho[j][k]+=rho[i][j][k]/nx;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout<<"# Computation time on CPU is "<<elapsed.count()<<" seconds."<<std::endl;
    std::cout<<"# mean at (iy, iz) = (0, 0): "<<meanRho[0][0]<<std::endl;
    std::cout<<"# mean at (iy, iz) = (0, 1): "<<meanRho[0][1]<<std::endl;
    std::cout<<"# mean at (iy, iz) = (0, 2): "<<meanRho[0][2]<<std::endl;
    std::cout<<"# mean at (iy, iz) = (0, 3): "<<meanRho[0][3]<<std::endl;
    
    return 0;
}