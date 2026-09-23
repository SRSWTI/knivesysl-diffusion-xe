#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace transpose_demo {

constexpr int tile_size = 32;
constexpr int block_rows = 8;

inline void cuda_check(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
    }
}

struct Layout {
    int rows;
    int columns;
    int stride;

    void validate() const {
        if (rows < 0 || columns < 0 || stride < columns) {
            throw std::invalid_argument("invalid matrix layout");
        }
    }
    std::size_t elements() const {
        validate();
        return std::size_t(rows) * stride;
    }
    Layout transposed(int padding = 0) const {
        if (padding < 0) throw std::invalid_argument("negative padding");
        return {columns, rows, rows + padding};
    }
};

// Input and output are distinct row-major matrices, with strides in elements.
// A 32x8 thread block processes a 32x32 tile, including partial edge tiles.
__global__ void transpose_tiled(const float* input, float* output,
                                int rows, int columns, int input_stride, int output_stride) {
    __shared__ float tile[tile_size][tile_size + 1];
    const int x = blockIdx.x * tile_size + threadIdx.x;
    const int y = blockIdx.y * tile_size + threadIdx.y;
    for (int j = 0; j < tile_size; j += block_rows) {
        if (x < columns && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[std::size_t(y + j) * input_stride + x];
        }
    }
    __syncthreads();

    const int output_x = blockIdx.x * tile_size + threadIdx.x;
    const int output_y = blockIdx.y * tile_size + threadIdx.y;
    for (int j = 0; j < tile_size; j += block_rows) {
        if (output_x < rows && output_y + j < columns) {
            output[std::size_t(output_y + j) * output_stride + output_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

inline void launch_transpose(const float* input, float* output, Layout source,
                             Layout destination, cudaStream_t stream = nullptr) {
    source.validate();
    destination.validate();
    if (destination.rows != source.columns || destination.columns != source.rows) {
        throw std::invalid_argument("destination is not the transposed shape");
    }
    if (!source.rows || !source.columns) return;
    if (!input || !output || input == output) {
        throw std::invalid_argument("transpose requires distinct non-null buffers");
    }
    const dim3 threads(tile_size, block_rows);
    const dim3 blocks((source.columns + tile_size - 1) / tile_size,
                      (source.rows + tile_size - 1) / tile_size);
    transpose_tiled<<<blocks, threads, 0, stream>>>(
        input, output, source.rows, source.columns, source.stride, destination.stride);
    cuda_check(cudaGetLastError(), "transpose launch");
}

class DeviceMatrix {
public:
    explicit DeviceMatrix(Layout layout) : layout_(layout) {
        if (layout_.elements()) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), layout_.elements() * sizeof(float)), "allocate matrix");
        }
    }
    ~DeviceMatrix() { if (data_) cudaFree(data_); }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    Layout layout() const noexcept { return layout_; }

    void upload(const std::vector<float>& source) {
        if (source.size() != layout_.elements()) throw std::invalid_argument("upload length mismatch");
        if (!source.empty()) {
            cuda_check(cudaMemcpy(data_, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice), "upload matrix");
        }
    }
    std::vector<float> download() const {
        std::vector<float> result(layout_.elements());
        if (!result.empty()) {
            cuda_check(cudaMemcpy(result.data(), data_, result.size() * sizeof(float), cudaMemcpyDeviceToHost), "download matrix");
        }
        return result;
    }

private:
    Layout layout_;
    float* data_ = nullptr;
};

class TransposePlan {
public:
    TransposePlan(Layout source, Layout destination) : source_(source), destination_(destination) {
        source_.validate();
        destination_.validate();
        if (source_.rows != destination_.columns || source_.columns != destination_.rows) {
            throw std::invalid_argument("incompatible transpose plan");
        }
    }
    void execute(const DeviceMatrix& input, DeviceMatrix& output, cudaStream_t stream = nullptr) const {
        const auto a = input.layout();
        const auto b = output.layout();
        if (a.rows != source_.rows || a.columns != source_.columns || a.stride != source_.stride ||
            b.rows != destination_.rows || b.columns != destination_.columns || b.stride != destination_.stride) {
            throw std::invalid_argument("buffer layout differs from plan");
        }
        launch_transpose(input.data(), output.data(), source_, destination_, stream);
    }
    std::size_t bytes_read() const noexcept {
        return std::size_t(source_.rows) * source_.columns * sizeof(float);
    }
    std::size_t bytes_written() const noexcept { return bytes_read(); }

private:
    Layout source_;
    Layout destination_;
};

} // namespace transpose_demo
