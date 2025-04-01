#ifndef FILE_READER_HPP
#define FILE_READER_HPP

#include <vector>
#include <string>

std::vector<std::vector<std::vector<float>>> readChunk(const std::string&, int, bool);
std::vector<std::vector<std::vector<float>>> loadStick(const int, const std::string&, bool);
void processRTV(std::vector<std::vector<std::vector<float>>>&, 
                std::vector<std::vector<std::vector<float>>>&, 
                std::vector<std::vector<std::vector<float>>>&, 
                std::vector<std::vector<std::vector<float>>>&, 
                const int, const int, const int) ;
std::vector<float> readBinaryFile(const std::string&, int&, int&, int&);
std::vector<float> readBinaryFileHalf(const std::string&, int&, int&, int&, bool);
std::vector<std::vector<std::vector<float>>> reshape(const std::vector<float>&, int, int, int);
void printRamUsage();

#endif
