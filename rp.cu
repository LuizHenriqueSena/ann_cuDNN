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

struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}
};



int main() {
	int num_gpus;
	std::vector<uint8_t> dataset(tamanho_img*amostras);
	dataset = {1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,
1,0,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,
1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,
1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,
1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,
1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,
1,0,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,
1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,
1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,
1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,
1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,
1,0,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,
1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,
1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,
1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,
1,1,1,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,
1,0,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,
1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,
1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,
1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,
1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,
1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,
1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,
1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,
1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

    	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	printf("valor da contagem %d \n", num_gpus);
	FullyConnectedLayer l1(25, 5), l2(5,6), l3(6,5);

	std::random_device rd;
	
	float wfc1 = sqrt(3.0f / (l1.inputs * l1.outputs));
	printf("valor do aleatorio %.6f \n", wfc1);
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);

	float wfc2 = sqrt(3.0f / (l2.inputs * l2.outputs));
	printf("valor do aleatorio %.6f \n", wfc2);
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

	float wfc3 = sqrt(3.0f / (l3.inputs * l3.outputs));
	printf("valor do aleatorio %.6f \n", wfc3);
        std::uniform_real_distribution<> dfc3(-wfc3, wfc3);
	
        for (auto&& iter : l1.pneurons)
            iter = static_cast<float>(0.5);
        for (auto&& iter : l1.pbias)
            iter = static_cast<float>(0.5);
        for (auto&& iter : l2.pneurons)
            iter = static_cast<float>(0.5);
        for (auto&& iter : l2.pbias)
            iter = static_cast<float>(0.5);
        for (auto&& iter : l3.pneurons)
            iter = static_cast<float>(0.5);
        for (auto&& iter : l3.pbias)
            iter = static_cast<float>(0.5);


	
		

}
