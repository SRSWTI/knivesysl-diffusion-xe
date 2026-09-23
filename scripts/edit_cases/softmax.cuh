#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace softmax_demo {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&stream_), "create stream"); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { check_cuda(cudaStreamSynchronize(stream_), "stream synchronize"); }
private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_) check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "allocate");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    void upload(const std::vector<T>& host) {
        if (host.size() != count_) throw std::invalid_argument("upload size mismatch");
        if (count_) check_cuda(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_) check_cuda(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return host;
    }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct MatrixShape {
    int rows = 0;
    int columns = 0;
    int stride = 0;

    void validate() const {
        if (rows < 0 || columns < 0 || stride < columns) {
            throw std::invalid_argument("invalid matrix shape");
        }
    }
    std::size_t storage_size() const {
        validate();
        return std::size_t(rows) * std::size_t(stride);
    }
};

struct InputView {
    const float* data;
    MatrixShape shape;
};

struct OutputView {
    float* data;
    MatrixShape shape;
};

// A block owns one row. The public launcher always uses 256 threads.
// Inputs are finite floats. Row padding belongs to the caller and is not output.
__global__ void row_softmax_kernel(const float* input, float* output,
                                   int rows, int columns,
                                   int input_stride, int output_stride) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    __shared__ float sums[256];

    const float value = lane < columns
        ? expf(input[std::size_t(row) * input_stride + lane]) : 0.0f;
    sums[lane] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (lane < step) sums[lane] += sums[lane + step];
        __syncthreads();
    }
    if (lane < columns) {
        output[std::size_t(row) * output_stride + lane] = value / sums[0];
    }
}

inline void launch_row_softmax(InputView input, OutputView output, cudaStream_t stream = nullptr) {
    input.shape.validate();
    output.shape.validate();
    if (input.shape.rows != output.shape.rows || input.shape.columns != output.shape.columns) {
        throw std::invalid_argument("input and output dimensions differ");
    }
    if (input.shape.rows == 0 || input.shape.columns == 0) return;
    if (!input.data || !output.data) throw std::invalid_argument("null matrix data");
    row_softmax_kernel<<<input.shape.rows, 256, 0, stream>>>(
        input.data, output.data, input.shape.rows, input.shape.columns,
        input.shape.stride, output.shape.stride);
    check_cuda(cudaGetLastError(), "launch softmax");
}

class SoftmaxPlan {
public:
    explicit SoftmaxPlan(MatrixShape input, int output_stride)
        : input_(input), output_{input.rows, input.columns, output_stride} {
        input_.validate();
        output_.validate();
    }

    void execute(const float* input, float* output, cudaStream_t stream = nullptr) const {
        launch_row_softmax({input, input_}, {output, output_}, stream);
    }

    std::size_t input_elements() const { return input_.storage_size(); }
    std::size_t output_elements() const { return output_.storage_size(); }
    int rows() const noexcept { return input_.rows; }
    int columns() const noexcept { return input_.columns; }

private:
    MatrixShape input_;
    MatrixShape output_;
};

class HostMatrix {
public:
    explicit HostMatrix(MatrixShape shape, float initial_value = 0.0f)
        : shape_(shape), storage_(shape.storage_size(), initial_value) {}

    float& at(int row, int column) {
        bounds_check(row, column);
        return storage_[std::size_t(row) * shape_.stride + column];
    }
    float at(int row, int column) const {
        bounds_check(row, column);
        return storage_[std::size_t(row) * shape_.stride + column];
    }
    const std::vector<float>& storage() const noexcept { return storage_; }
    MatrixShape shape() const noexcept { return shape_; }
    void assign_storage(std::vector<float> storage) {
        if (storage.size() != storage_.size()) throw std::invalid_argument("storage size mismatch");
        storage_.swap(storage);
    }
    double row_sum(int row) const {
        double sum = 0.0;
        for (int column = 0; column < shape_.columns; ++column) sum += at(row, column);
        return sum;
    }

private:
    void bounds_check(int row, int column) const {
        if (row < 0 || row >= shape_.rows || column < 0 || column >= shape_.columns) {
            throw std::out_of_range("matrix index out of range");
        }
    }
    MatrixShape shape_;
    std::vector<float> storage_;
};

} // namespace softmax_demo
