#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void add_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = a[i] + b[i]; }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = a[i] - b[i]; }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = a[i] * b[i]; }
}

__global__ void div_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = a[i] / (b[i] + 1e-8f); } // Epsilon prevents DivByZero
}

__global__ void pow_kernel(const float* a, float exponent, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = powf(a[i], exponent); }
}

__global__ void neg_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = -a[i]; }
}

__global__ void relu_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = fmaxf(0.0f, a[i]); }
}

__global__ void gt_zero_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = a[i] > 0.0f ? 1.0f : 0.0f; }
}

__global__ void sigmoid_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = 1.0f / (1.0f + expf(-a[i])); }
}

__global__ void tanh_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = tanhf(a[i]); }
}

__global__ void fill_kernel(float* out, float val, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) { out[i] = val; }
}

__global__ void sum_kernel(const float* a, float* out, int size) {
    CUDA_1D_KERNEL_LOOP(i, size) {
        atomicAdd(out, a[i]); 
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < rows && c < cols) {
        output[c * rows + r] = input[r * cols + c];
    }
}


int get_blocks(int size, int threads) {
    return (size + threads - 1) / threads;
}

void launch_add(const float* a, const float* b, float* out, int size) { add_kernel<<<get_blocks(size, 256), 256>>>(a, b, out, size); }
void launch_sub(const float* a, const float* b, float* out, int size) { sub_kernel<<<get_blocks(size, 256), 256>>>(a, b, out, size); }
void launch_mul(const float* a, const float* b, float* out, int size) { mul_kernel<<<get_blocks(size, 256), 256>>>(a, b, out, size); }
void launch_div(const float* a, const float* b, float* out, int size) { div_kernel<<<get_blocks(size, 256), 256>>>(a, b, out, size); }
void launch_pow(const float* a, float exp, float* out, int size) { pow_kernel<<<get_blocks(size, 256), 256>>>(a, exp, out, size); }
void launch_neg(const float* a, float* out, int size) { neg_kernel<<<get_blocks(size, 256), 256>>>(a, out, size); }
void launch_relu(const float* a, float* out, int size) { relu_kernel<<<get_blocks(size, 256), 256>>>(a, out, size); }
void launch_gt_zero(const float* a, float* out, int size) { gt_zero_kernel<<<get_blocks(size, 256), 256>>>(a, out, size); }
void launch_sigmoid(const float* a, float* out, int size) { sigmoid_kernel<<<get_blocks(size, 256), 256>>>(a, out, size); }
void launch_tanh(const float* a, float* out, int size) { tanh_kernel<<<get_blocks(size, 256), 256>>>(a, out, size); }
void launch_fill(float* out, float val, int size) { fill_kernel<<<get_blocks(size, 256), 256>>>(out, val, size); }

void launch_sum(const float* a, float* out, int size) {
    fill_kernel<<<1, 1>>>(out, 0.0f, 1); // Zero out the result memory first
    sum_kernel<<<get_blocks(size, 256), 256>>>(a, out, size);
}

void launch_matmul(const float* a, const float* b, float* out, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_kernel<<<blocks, threads>>>(a, b, out, M, N, K);
}

void launch_transpose(const float* a, float* out, int rows, int cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    transpose_kernel<<<blocks, threads>>>(a, out, rows, cols);
}