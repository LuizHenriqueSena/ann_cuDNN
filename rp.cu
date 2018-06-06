#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define tamanho_img 25
#define amostras 200

int main() {
	int num_gpus;
	std::vector<uint8_t> dataset(tamanho_img*amostras);
    	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	printf("valor da contagem %d \n", num_gpus);

	

	
}
