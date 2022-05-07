#include<iostream>

int main()
{
    //定义一个cuda的设备属性结构体
    cudaDeviceProp prop;
    //获取第1个gpu设备的属性信息
    cudaGetDeviceProperties(&prop, 0);
    //每个block的最大线程数
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    //block的维度
    for (int i = 0; i < 3; ++i) std::cout << "maxThreadsDim[" << i << "]: " << prop.maxThreadsDim[i] << std::endl;
    //输出最大的gridSize
    std::cout << std::endl;
    for (int i = 0; i < 3; ++i) std::cout << "maxGridSize[" << i << "]: " << prop.maxGridSize[i] << std::endl;
    return 0;
}