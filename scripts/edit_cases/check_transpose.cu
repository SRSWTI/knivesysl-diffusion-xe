#include "subject.cuh"
#include <iostream>
#include <cstdlib>
using namespace transpose_demo;
static int checks=0;
static void require(bool condition,const char* label) {
    ++checks;
    if(!condition) { std::cerr << "FAIL " << label << " at check " << checks << '\n'; std::exit(1); }
}
int main() {
    try {
        int devices=0;
        cuda_check(cudaGetDeviceCount(&devices),"CUDA availability");
        require(devices>0,"CUDA device exists");
        for(auto dimensions:std::vector<std::pair<int,int>>{{1,1},{1,65},{65,1},{7,13},{31,33},{32,32},{33,31},{63,95},{65,97},{257,513}}) {
            for(int padding:{0,5}) {
                Layout source{dimensions.first,dimensions.second,dimensions.second+padding};
                Layout destination=source.transposed(padding+3);
                std::vector<float> original(source.elements(),-7777.0f);
                for(int row=0;row<source.rows;++row) for(int col=0;col<source.columns;++col)
                    original[std::size_t(row)*source.stride+col]=float(row*1024+col);
                DeviceMatrix input(source),output(destination);
                input.upload(original);
                output.upload(std::vector<float>(destination.elements(),-9999.0f));
                TransposePlan plan(source,destination);
                plan.execute(input,output);
                cuda_check(cudaDeviceSynchronize(),"transpose synchronize");
                const auto actual=output.download();
                require(input.download()==original,"source remains unchanged");
                for(int row=0;row<destination.rows;++row) {
                    for(int col=0;col<destination.columns;++col)
                        require(actual[std::size_t(row)*destination.stride+col]==original[std::size_t(col)*source.stride+row],"rectangular tiled transpose reference agreement");
                    for(int col=destination.columns;col<destination.stride;++col)
                        require(actual[std::size_t(row)*destination.stride+col]==-9999.0f,"destination padding remains unchanged");
                }
            }
        }
        launch_transpose(nullptr,nullptr,{0,7,7},{7,0,0});
        std::cout << "PASS " << checks << " CUDA transpose checks\n";
    } catch(const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 2; }
}
