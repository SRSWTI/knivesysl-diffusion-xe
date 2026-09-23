#include "subject.cuh"
#include <iostream>
#include <cstdlib>
using namespace softmax_demo;
static int checks = 0;
static void require(bool condition, const char* label) {
    ++checks;
    if (!condition) { std::cerr << "FAIL " << label << " at check " << checks << '\n'; std::exit(1); }
}
int main() {
    try {
        int devices = 0;
        check_cuda(cudaGetDeviceCount(&devices), "CUDA availability");
        require(devices>0,"CUDA device exists");
        for (int columns : {1,7,31,32,33,127,255,256,257,513,1025,4097}) {
            for (float offset : {0.0f,1000.0f,-1000.0f}) {
                const int rows = 5;
                MatrixShape input_shape{rows,columns,columns+3}, output_shape{rows,columns,columns+7};
                HostMatrix input(input_shape,99999.0f);
                for (int row=0;row<rows;++row) for(int col=0;col<columns;++col)
                    input.at(row,col)=offset+float((row*17+col*13)%41-20)*0.7f;
                DeviceBuffer<float> gpu_input(input_shape.storage_size()), gpu_output(output_shape.storage_size());
                gpu_input.upload(input.storage());
                gpu_output.upload(std::vector<float>(output_shape.storage_size(),-12345.0f));
                SoftmaxPlan plan(input_shape,output_shape.stride);
                Stream stream;
                plan.execute(gpu_input.data(),gpu_output.data(),stream.get());
                stream.synchronize();
                const auto output=gpu_output.download();
                require(gpu_input.download()==input.storage(),"input and padding unchanged");
                for(int row=0;row<rows;++row) {
                    double maximum=-std::numeric_limits<double>::infinity();
                    for(int col=0;col<columns;++col) maximum=std::max(maximum,double(input.at(row,col)));
                    double denominator=0, observed_sum=0;
                    for(int col=0;col<columns;++col) denominator+=std::exp(double(input.at(row,col))-maximum);
                    for(int col=0;col<columns;++col) {
                        const double expected=std::exp(double(input.at(row,col))-maximum)/denominator;
                        const float actual=output[std::size_t(row)*output_shape.stride+col];
                        require(std::isfinite(actual),"finite result for finite extreme logits");
                        require(std::abs(actual-expected)<=3e-5+2e-4*expected,"stable softmax reference agreement");
                        observed_sum+=actual;
                    }
                    require(std::abs(observed_sum-1)<3e-5,"row normalizes to one");
                    for(int col=columns;col<output_shape.stride;++col)
                        require(output[std::size_t(row)*output_shape.stride+col]==-12345.0f,"output padding preserved");
                }
            }
        }
        launch_row_softmax({nullptr,{0,7,7}},{nullptr,{0,7,7}});
        check_cuda(cudaDeviceSynchronize(),"final synchronize");
        std::cout << "PASS " << checks << " CUDA softmax checks\n";
    } catch(const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 2; }
}
