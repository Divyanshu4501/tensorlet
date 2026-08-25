#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

void launch_add(const float* a, const float* b, float* out, int size);
void launch_sub(const float* a, const float* b, float* out, int size);
void launch_mul(const float* a, const float* b, float* out, int size);
void launch_div(const float* a, const float* b, float* out, int size);
void launch_pow(const float* a, float exp, float* out, int size);
void launch_neg(const float* a, float* out, int size);
void launch_relu(const float* a, float* out, int size);
void launch_gt_zero(const float* a, float* out, int size);
void launch_sigmoid(const float* a, float* out, int size);
void launch_tanh(const float* a, float* out, int size);
void launch_fill(float* out, float val, int size);
void launch_sum(const float* a, float* out, int size);
void launch_matmul(const float* a, const float* b, float* out, int M, int N, int K);
void launch_transpose(const float* a, float* out, int rows, int cols);


class GPUTensor {
public:
    float* data;
    size_t size;
    std::vector<ssize_t> shape;

    GPUTensor(py::array_t<float> input) {
        py::buffer_info buf = input.request();
        size = buf.size;
        shape = buf.shape;
        cudaMalloc(&data, size * sizeof(float));
        cudaMemcpy(data, buf.ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    GPUTensor(size_t size, std::vector<ssize_t> shape) : size(size), shape(shape) {
        cudaMalloc(&data, size * sizeof(float));
    }

    ~GPUTensor() {
        cudaFree(data);
    }

    py::array_t<float> to_cpu() {
        auto result = py::array_t<float>(shape);
        py::buffer_info buf = result.request();
        cudaMemcpy(buf.ptr, data, size * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
};

GPUTensor* add_tensors(GPUTensor& a, GPUTensor& b) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_add(a.data, b.data, out->data, a.size);
    return out;
}

GPUTensor* sub_tensors(GPUTensor& a, GPUTensor& b) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_sub(a.data, b.data, out->data, a.size);
    return out;
}

GPUTensor* mul_tensors(GPUTensor& a, GPUTensor& b) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_mul(a.data, b.data, out->data, a.size);
    return out;
}

GPUTensor* div_tensors(GPUTensor& a, GPUTensor& b) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_div(a.data, b.data, out->data, a.size);
    return out;
}

GPUTensor* pow_tensor(GPUTensor& a, float exponent) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_pow(a.data, exponent, out->data, a.size);
    return out;
}

GPUTensor* neg_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_neg(a.data, out->data, a.size);
    return out;
}

GPUTensor* relu_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_relu(a.data, out->data, a.size);
    return out;
}

GPUTensor* gt_zero_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_gt_zero(a.data, out->data, a.size);
    return out;
}

GPUTensor* sigmoid_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_sigmoid(a.data, out->data, a.size);
    return out;
}

GPUTensor* tanh_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_tanh(a.data, out->data, a.size);
    return out;
}

GPUTensor* ones_like(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_fill(out->data, 1.0f, a.size);
    return out;
}

GPUTensor* zeros_like(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(a.size, a.shape);
    launch_fill(out->data, 0.0f, a.size);
    return out;
}

GPUTensor* sum_tensor(GPUTensor& a) {
    GPUTensor* out = new GPUTensor(1, {1});
    launch_sum(a.data, out->data, a.size);
    return out;
}

GPUTensor* mean_tensor(GPUTensor& a) {
    GPUTensor* sum_val = sum_tensor(a);
    GPUTensor* size_tensor = new GPUTensor(1, {1});
    launch_fill(size_tensor->data, static_cast<float>(a.size), 1);
    
    GPUTensor* out = div_tensors(*sum_val, *size_tensor);
    delete sum_val;
    delete size_tensor;
    return out;
}

GPUTensor* reshape_tensor(GPUTensor& a, std::vector<ssize_t> new_shape) {
    GPUTensor* out = new GPUTensor(a.size, new_shape);
    cudaMemcpy(out->data, a.data, a.size * sizeof(float), cudaMemcpyDeviceToDevice);
    return out;
}

GPUTensor* matmul_tensors(GPUTensor& a, GPUTensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::runtime_error("Matmul requires 2D tensors");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::runtime_error("Matrix inner dimensions must agree");
    }

    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];

    GPUTensor* out = new GPUTensor(M * N, {M, N});
    launch_matmul(a.data, b.data, out->data, M, N, K);
    return out;
}

GPUTensor* transpose_tensor(GPUTensor& a) {
    if (a.shape.size() != 2) {
        throw std::runtime_error("Transpose currently only supports 2D tensors");
    }

    int rows = a.shape[0];
    int cols = a.shape[1];

    GPUTensor* out = new GPUTensor(rows * cols, {cols, rows});
    launch_transpose(a.data, out->data, rows, cols);
    return out;
}

PYBIND11_MODULE(tensorlet_cuda, m) {
    py::class_<GPUTensor>(m, "GPUTensor")
        .def("to_cpu", &GPUTensor::to_cpu)
        .def_property_readonly("shape", [](const GPUTensor& t) { return t.shape; })
        .def_property_readonly("size", [](const GPUTensor& t) { return t.size; });

    m.def("to_device", [](py::array_t<float> arr) { return new GPUTensor(arr); });
    m.def("to_cpu", [](GPUTensor& t) { return t.to_cpu(); });
    
    m.def("add", &add_tensors);
    m.def("sub", &sub_tensors);
    m.def("mul", &mul_tensors);
    m.def("div", &div_tensors);
    m.def("pow", &pow_tensor);
    
    m.def("neg", &neg_tensor);
    m.def("relu", &relu_tensor);
    m.def("sigmoid", &sigmoid_tensor);
    m.def("tanh", &tanh_tensor);
    m.def("greater_than_zero", &gt_zero_tensor);
    
    m.def("sum", &sum_tensor);
    m.def("mean", &mean_tensor);
    
    m.def("matmul", &matmul_tensors);
    m.def("transpose", &transpose_tensor);
    m.def("reshape", &reshape_tensor);
    
    m.def("ones_like", &ones_like);
    m.def("zeros_like", &zeros_like);
}