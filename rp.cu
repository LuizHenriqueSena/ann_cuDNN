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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>


// Block width for CUDA kernels
#define BW 128

#define tamanho_img 25
#define amostras 200
#define train_size 200

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


/**
 * Computes ceil(x / y) for integral nonnegative values.
 *teste
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

float funcaoDeAtivacao(float u) {
	float retorno;
	retorno = (1/(1 + pow(2.718281,(u*(-1)))));
	return retorno;
}

float derivadaFuncaoDeAtivacao(float u) {
	float retorno;
	retorno = funcaoDeAtivacao(u)*(1 - funcaoDeAtivacao(u));
	return retorno;
}

__global__ void SigmoidBackprop(float * label, float * outputLayer, float * lastInputLayer, int size, ) {
	
	
}

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

void imprimeSaidas(float *data, int tamanho){
	int i = 0;
	for(i = 0; i <tamanho; i++) {
		printf("Saida[%d] = %.6f \n", i, data[i]);
	}
}
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
        pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

struct TrainingContext
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, l1Tensor, l2Tensor, l3Tensor;
    cudnnActivationDescriptor_t l1Activation, l2Activation, l3Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_l1, &ref_l2, &ref_l3;

    // Disable copying
    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;

    TrainingContext(int gpuid, int batch_size,
                    FullyConnectedLayer& l1, FullyConnectedLayer& l2, FullyConnectedLayer& l3) : ref_l1(l1), ref_l2(l2), ref_l3(l3), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;

        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&l1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&l2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&l3Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&l1Activation));
        checkCUDNN(cudnnCreateActivationDescriptor(&l2Activation));
        checkCUDNN(cudnnCreateActivationDescriptor(&l3Activation));

        
        // Set tensor descriptor sizes

        checkCUDNN(cudnnSetTensor4dDescriptor(l1Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, l1.outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(l2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, l2.outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(l3Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, l3.outputs, 1, 1));

        checkCUDNN(cudnnSetActivationDescriptor(l1Activation, CUDNN_ACTIVATION_SIGMOID,
                                                CUDNN_PROPAGATE_NAN, 0.0));

	checkCUDNN(cudnnSetActivationDescriptor(l2Activation, CUDNN_ACTIVATION_SIGMOID,
                                                CUDNN_PROPAGATE_NAN, 0.0));

	checkCUDNN(cudnnSetActivationDescriptor(l3Activation, CUDNN_ACTIVATION_SIGMOID,
                                                CUDNN_PROPAGATE_NAN, 0.0));

    }

    ~TrainingContext()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(l1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(l2Tensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(l3Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(l1Activation));
        checkCUDNN(cudnnDestroyActivationDescriptor(l2Activation));
    }


    void ForwardPropagation(float *data, float *fc1, float *fc1relu,
                            float *fc2, float *fc2relu, float *fc3, float *result,
                            float *pfc1, float *pfc1bias,
                            float *pfc2, float *pfc2bias,
                            float *pfc3, float *pfc3bias, float *onevec)
    {        
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));


        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*data)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_l1.outputs, m_batchSize, ref_l1.inputs,
                                    &alpha,
                                    pfc1, ref_l1.inputs,
                                    data, ref_l1.inputs,
                                    &beta,
                                    fc1, ref_l1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_l1.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc1bias, ref_l1.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_l1.outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, l1Activation, &alpha,
                                          l1Tensor, fc1, &beta, l1Tensor, fc1relu));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_l2.outputs, m_batchSize, ref_l2.inputs,
                                    &alpha,
                                    pfc2, ref_l2.inputs,
                                    fc1relu, ref_l2.inputs,
                                    &beta,
                                    fc2, ref_l2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_l2.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc2bias, ref_l2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_l2.outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, l2Activation, &alpha,
                                          l2Tensor, fc2, &beta, l2Tensor, fc2relu));

        // Forward propagate neurons using weights (fc3 = pfc3'*fc2relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_l3.outputs, m_batchSize, ref_l3.inputs,
                                    &alpha,
                                    pfc3, ref_l3.inputs,
                                    fc2relu, ref_l3.inputs,
                                    &beta,
                                    fc3, ref_l2.outputs));
        // Add bias using GEMM's "beta" (fc3 += pfc3bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_l3.outputs, m_batchSize, 1,
                                    &alpha,
                                    pfc3bias, ref_l3.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc3, ref_l3.outputs));



        // Softmax loss
        //checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        //                               &alpha, l3Tensor, fc3, &beta, l3Tensor, result));
	checkCUDNN(cudnnActivationForward(cudnnHandle, l3Activation, &alpha,
                                          l3Tensor, fc3, &beta, l3Tensor, result));
    }

    void Backpropagation(float *data, float *labels, float *fc1, float *fc1relu,
                         float *fc2, float *fc2relu, float *fc3, float *fc3sfmx, float *dloss_data,
                         float *pfc1, float *pfc1bias,
                         float *pfc2, float *pfc2bias,
			 float *pfc3, float *pfc3bias,
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                         float *gfc2, float *gfc2bias, float *dfc2, float *dfc2relu,
			 float *gfc3, float *gfc3bias, float *dfc3,
                         float *onevec)
    {    
        float alpha = 1.0f, beta = 0.0f;

        float scalVal = 1.0f / static_cast<float>(m_batchSize);

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkCudaErrors(cudaMemcpyAsync(dloss_data, fc3sfmx, sizeof(float) * m_batchSize * ref_l3.outputs, cudaMemcpyDeviceToDevice));
        
        // Accounting for batch size in SGD
        //checkCudaErrors(cublasSscal(cublasHandle, ref_l3.outputs * m_batchSize, &scalVal, dloss_data, 1));

	 // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, l3Activation, &alpha,
                                           l3Tensor, fc3sfmx, l3Tensor, labels,
                                           l3Tensor, fc3, &beta, l3Tensor, dloss_data));

        // FC3 layer
        // Compute derivative with respect to weights: gfc3 = (fc2relu * dfc3smax')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_l3.inputs, ref_l3.outputs, m_batchSize,
                                    &alpha, fc2relu, ref_l3.inputs, dloss_data, ref_l3.outputs, &beta, gfc3, ref_l3.inputs));
        // Compute derivative with respect to bias: gfc3bias = dfc3smax * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_l3.outputs, m_batchSize,
                                    &alpha, dloss_data, ref_l3.outputs, onevec, 1, &beta, gfc3bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc3*dfc3smax (500x10*10xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_l3.inputs, m_batchSize, ref_l3.outputs,
                                    &alpha, pfc3, ref_l3.inputs, dloss_data, ref_l3.outputs, &beta, dfc3, ref_l3.inputs));
        
        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, l2Activation, &alpha,
                                           l2Tensor, fc2relu, l2Tensor, dfc3,
                                           l2Tensor, fc2, &beta, l2Tensor, dfc2relu));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_l2.inputs, ref_l2.outputs, m_batchSize,
                                    &alpha, fc1relu, ref_l2.inputs, dfc2relu, ref_l2.outputs, &beta, gfc2, ref_l2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_l2.outputs, m_batchSize,
                                    &alpha, dfc2relu, ref_l2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_l2.inputs, m_batchSize, ref_l2.outputs,
                                    &alpha, pfc2, ref_l2.inputs, dfc2relu, ref_l2.outputs, &beta, dfc2, ref_l2.inputs));

	        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, l1Activation, &alpha,
                                           l1Tensor, fc1relu, l1Tensor, dfc2,
                                           l1Tensor, fc1, &beta, l1Tensor, dfc1relu));


	 // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (data * dfc1relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_l1.inputs, ref_l1.outputs, m_batchSize,
                                    &alpha, data, ref_l1.inputs, dfc1relu, ref_l1.outputs, &beta, gfc1, ref_l1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_l1.outputs, m_batchSize,
                                    &alpha, dfc1relu, ref_l1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_l1.inputs, m_batchSize, ref_l1.outputs,
                                    &alpha, pfc1, ref_l1.inputs, dfc1relu, ref_l1.outputs, &beta, dfc1, ref_l1.inputs));



    }

    void UpdateWeights(float learning_rate,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
			float *pfc3, float *pfc3bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias,
			float *gfc3, float *gfc3bias)
    {    
        float alpha = -learning_rate;

        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Fully connected 1
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l1.pneurons.size()),
                                    &alpha, gfc1, 1, pfc1, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l1.pbias.size()),
                                    &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l2.pneurons.size()),
                                    &alpha, gfc2, 1, pfc2, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l2.pbias.size()),
                                    &alpha, gfc2bias, 1, pfc2bias, 1));

	// Fully connected 3
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l3.pneurons.size()),
                                    &alpha, gfc3, 1, pfc3, 1));
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_l3.pbias.size()),
                                    &alpha, gfc3bias, 1, pfc3bias, 1));
    }



};




int main() {
	int num_gpus;
	std::vector<float> dataset(tamanho_img*amostras);
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
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,
1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,
1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,
1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

	std::vector<uint8_t> saidas(amostras*5);
	saidas = {1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,1,0,0,0,0
,0,1,0,0,0
,0,0,1,0,0
,0,0,0,1,0
,0,0,0,0,1
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0
,0,0,0,0,0};

//1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
//	1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
//	1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
//	1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,
//	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

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

	TrainingContext context(0, 1, l1, l2, l3);



float *d_data, *d_labels, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2relu, *d_fc3, *d_fc3smax;

	checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * 25));
	checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 5));
	checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * l1.outputs));    
	checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * l1.outputs));
	checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * l2.outputs));
	checkCudaErrors(cudaMalloc(&d_fc2relu, sizeof(float) * context.m_batchSize * l2.outputs));
	checkCudaErrors(cudaMalloc(&d_fc3,     sizeof(float) * context.m_batchSize * l3.outputs));
	checkCudaErrors(cudaMalloc(&d_fc3smax, sizeof(float) * context.m_batchSize * l3.outputs));   
	
    // Network parameters
    float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias, *d_pfc3, *d_pfc3bias;
		
    checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * l1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * l1.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * l2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * l2.pbias.size())); 
    checkCudaErrors(cudaMalloc(&d_pfc3,       sizeof(float) * l3.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_pfc3bias,   sizeof(float) * l3.pbias.size()));  

    // Network parameter gradients
    float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias, *d_gfc3, *d_gfc3bias;
    
    checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * l1.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * l1.pbias.size()));    
    checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * l2.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * l2.pbias.size()));
    checkCudaErrors(cudaMalloc(&d_gfc3,       sizeof(float) * l3.pneurons.size()));
    checkCudaErrors(cudaMalloc(&d_gfc3bias,   sizeof(float) * l3.pbias.size()));
    
    // Differentials w.r.t. data
    float *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2relu, *d_dfc3, *d_dfc3sfmax, *d_dlossdata;

    checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * l1.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * l1.outputs));
    checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * l2.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc2relu, sizeof(float) * context.m_batchSize * l2.outputs));
    checkCudaErrors(cudaMalloc(&d_dfc3,     sizeof(float) * context.m_batchSize * l3.inputs));
    checkCudaErrors(cudaMalloc(&d_dfc3sfmax, sizeof(float) * context.m_batchSize * l3.outputs));
    checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * l3.outputs));


    float *d_onevec;   
    checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));


    // Copy initial network to device
    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &l1.pneurons[0],      sizeof(float) * l1.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &l1.pbias[0],     sizeof(float) * l1.pbias.size(),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &l2.pneurons[0],      sizeof(float) * l2.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &l2.pbias[0],     sizeof(float) * l2.pbias.size(),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc3, &l3.pneurons[0],      sizeof(float) * l3.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc3bias, &l3.pbias[0],     sizeof(float) * l3.pbias.size(),    cudaMemcpyHostToDevice));
    
    // Fill one-vector with ones
    FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);


 checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 200; ++iter)
    {
        // Train
        int imageid = iter % (train_size / context.m_batchSize);

        // Prepare current batch on device
        checkCudaErrors(cudaMemcpyAsync(d_data, &dataset[imageid * context.m_batchSize*25],
                                        sizeof(float) * context.m_batchSize*25 , cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_labels, &saidas[imageid * context.m_batchSize*5],
                                        sizeof(float) * context.m_batchSize*5, cudaMemcpyHostToDevice));
        
        // Forward propagation
        context.ForwardPropagation(d_data, d_fc1, d_fc1relu, 
				   d_fc2, d_fc2relu, d_fc3, d_fc3smax,
                                   d_pfc1, d_pfc1bias, 
				   d_pfc2, d_pfc2bias, 
				   d_pfc3, d_pfc3bias, d_onevec);

        // Backward propagation
        context.Backpropagation( d_data, d_labels, d_fc1, d_fc1relu, 
				d_fc2, d_fc2relu, d_fc3, d_fc3smax, d_dlossdata,
                                d_pfc1, d_pfc1bias, 
				d_pfc2, d_pfc2bias,
				d_pfc3, d_pfc3bias,
                                d_gfc1, d_gfc1bias, d_dfc1, d_dfc1relu,
				d_gfc2, d_gfc2bias, d_dfc2, d_dfc2relu, 
				d_gfc3, d_gfc3bias, d_dfc3, 
				d_onevec);

        // Compute learning rate
	float learningRate = 0.5;
        //float learningRate = static_cast<float>(1 * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));

        // Update weights
        context.UpdateWeights(learningRate,
                              d_pfc1, d_pfc1bias, 
			      d_pfc2, d_pfc2bias,
			      d_pfc3, d_pfc3bias,
                              d_gfc1, d_gfc1bias,
			      d_gfc2, d_gfc2bias,
			      d_gfc3, d_gfc3bias);
    }
	
	std::vector<float> entradaTeste(25);
	entradaTeste = {1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1};
	std::vector<float> saidaAtual(l3.outputs);
	float *d_saidaAtual;

	checkCudaErrors(cudaMalloc(&d_saidaAtual, sizeof(float) * context.m_batchSize * l3.outputs)); 
	        checkCudaErrors(cudaMemcpyAsync(d_data, &entradaTeste[0],
                                        sizeof(float) * context.m_batchSize*25 , cudaMemcpyHostToDevice));
	        context.ForwardPropagation(d_data, d_fc1, d_fc1relu, 
				   d_fc2, d_fc2relu, d_fc3, d_saidaAtual,
                                   d_pfc1, d_pfc1bias, 
				   d_pfc2, d_pfc2bias, 
				   d_pfc3, d_pfc3bias, d_onevec);

	checkCudaErrors(cudaMemcpy(&saidaAtual[0], d_saidaAtual, sizeof(float) * l3.outputs, cudaMemcpyDeviceToHost));
	imprimeSaidas(&saidaAtual[0], 5);
	
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    //printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);



}
