# CalcLyaTauGpu

This repository contains C++/CUDA scripts for executing calculations on 3D grid data with GPU acceleration. 
Specifically, the code calculates Lyα opacity in a simulated universe, which will be measured by upcoming near-future astronomical surveys.

The calculation involves an immense number of simple operations, making GPUs an ideal choice for execution. 
I have confirmed that the implementation shared here is at least 100 times faster than a CPU-only calculation. 
I am sharing the code for those interested in performing similar computations.

Since the scripts are written in CUDA, an NVIDIA GPU and a compatible compiler (nvcc) are required to run the code.

## Repository contents

[1] main.sh: compilation and execution script

- Shows how to compile and execute the script.

- The executable main.exe takes two integer inputs, indicating different chunks of the entire 8192^3 mesh to be processed, allowing MPI parallelization.

[2] main.cpp: main driver 

[3] file_reader.cpp and file_reader.hpp: file reader

- Written for the CoDa simulations. Users are encouraged to edit these files for their own simulation format.

- 
