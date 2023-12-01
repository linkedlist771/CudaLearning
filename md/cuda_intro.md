C++ program that adds the elements of two arrays with a million elements each.
```c++
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}
```

```bash
Max error: 0
```

To run this code on GPU, add a `__global__` to the function , which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
```c++
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```

These `__global__` 
functions are known as kernels, and code 
that runs on the GPU is often called device
code, while code that runs on the CPU is host code.

- `unified memory`: Both GPU and CPU can access.

To allocate data in unified memory, call 
`cudaMallocManaged()`, which returns a 
pointer that you can access from host (CPU) code or device (GPU) code.
To free the data, just pass the pointer to `cudaFree()`.

```c++
  // Allocate Unified Memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  ...

  // Free memory
  cudaFree(x);
  cudaFree(y);
```

Finally, I need to launch the add() kernel, which invokes it on the GPU. CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>

```c++
add<<<1, 1>>>(N, x, y);

```

Just one more thing: I need the CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this I just call cudaDeviceSynchronize() before doing the final error checking on the CPU.

complete code:

```c++
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

>- Grid Size : This specifies the number of blocks that should be used in the grid for this kernel launch. In CUDA, a grid is a large group of blocks that can execute the same kernel code independently. A grid can be one-dimensional, two-dimensional, or three-dimensional. In your example, a grid size of 1 means that there is only one block in the grid.
>-  Block Size : This specifies the number of threads per block. In CUDA, a block is a group of threads that can cooperate together by sharing data through some fast shared memory and by synchronizing their execution to coordinate memory accesses. Like grids, blocks can also be one-dimensional, two-dimensional, or three-dimensional. In your example, a block size of 1 means that there is only one thread per block.


CUDA files have the file extension .cu. So save this code in a file called add.cu and compile it with nvcc, the CUDA C++ compiler.

```bash
nvcc add.cu -o add_cuda
./add_cuda
```


```bash
Max error: 0
```

This is only a first step, because as written, this kernel is only correct for a single thread, since every thread that runs it will perform the add on the whole array. Moreover, there is a race condition since multiple parallel threads would both read and write the same locations.


Also, `nvprof` can be used to check how long the kernel will run.

```bash
nvprof ./add_cuda
```

But it does not support for 8G or higher VRAM.

Turn this into `1 block 256 threads`:

```c++

#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
int index = threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < n; i += stride)
y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}

```

