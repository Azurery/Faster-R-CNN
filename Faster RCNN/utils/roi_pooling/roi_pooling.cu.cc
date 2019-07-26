#if GOOGLE_CUDA

#include <iostream>
#include <stdio.h>
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


#define index4_4(index, d1, d2, d3, d4) (index % d4)
#define index4_3(index, d1, d2, d3, d4) ((index / d4) % d3)

// cudaError_t是cuda 运行时API，任何cuda API运行都有可能发生错误，只有当正常运行后才返回cudaSuccess。
// 为什么专门写宏呢，为了避免cudaError_t重复定义，同时也是省事。
#define CUDA_CHECK(condition) \ 
    do { \
        cudaError_t error = condition; \
        if(error != cudaSuccess) { \
            return 1; \
        } \
    } while(0)

#define CUDA_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// 每个block使用512个线程
const int CAFFE_CUDA_NUM_THREADS = 512;




#define Dtype float

__global__ void RoiPoolingKernel(const Dtype* input, const int* rois,
                                int num_rois, int channels, int height, int width,
                                int pooling_height, int pooling_width,
                                Dtype* output, int* argmax_output) {
    
    int output_size = num_rois * channels * pooling_height * pooling_width;

    CUDA_KERNEL_LOOP(index, output_size) {
        // 因为维度的顺序是（N,C,H,W），第n个样本第c个通道的(h,w)坐标的索引 
        // index=(n-1)*C*H*W + (c-1)*H*W + (h-1)*W + w
        int 
    }

}
