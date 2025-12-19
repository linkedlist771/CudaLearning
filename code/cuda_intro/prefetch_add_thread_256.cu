

//
// Created by DingLi on 2023-12-01.
//
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ 
void add_kernel(int n, float *x, float *y)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i=index; i<n; i+=stride)
    {
        y[i] += x[i];
    }
}

int main(void)
{
    int N = 1<<20; // 1M elements

    float *x ;
    float *y;


    // Run kernel on 1M elements on the CPU
    // add(N, x, y);
    
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    int device_id=0;
    cudaStream_t stream = 0;  



    // initialize x and y arrays on the host after mannaged memory
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    cudaMemPrefetchAsync(x, N*sizeof(float), device_id, stream);
    cudaMemPrefetchAsync(y, N*sizeof(float), device_id, stream);

    int threads = 256;
    int blocks = (N+threads-1) / threads;

    add_kernel<<<blocks, threads>>>(N, x, y);

    cudaDeviceSynchronize(); // 必须同步

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory

    // 使用cuda后要cudaFree
    cudaFree(x);
    cudaFree(y);
    return 0;
}