export OMP_NUM_THREADS=192
nvcc -Xcompiler -fopenmp -o main sig_a_app.cpp checkGPUs.cpp constants.cpp file_reader.cpp calcTrGPU.cu main.cpp || exit
#&& ./main 0 0

for i in {0..63}; do 
    echo $i
    ./main $i 0
done
