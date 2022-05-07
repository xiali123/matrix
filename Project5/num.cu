#include<iostream>

int main()
{
    //����һ��cuda���豸���Խṹ��
    cudaDeviceProp prop;
    //��ȡ��1��gpu�豸��������Ϣ
    cudaGetDeviceProperties(&prop, 0);
    //ÿ��block������߳���
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    //block��ά��
    for (int i = 0; i < 3; ++i) std::cout << "maxThreadsDim[" << i << "]: " << prop.maxThreadsDim[i] << std::endl;
    //�������gridSize
    std::cout << std::endl;
    for (int i = 0; i < 3; ++i) std::cout << "maxGridSize[" << i << "]: " << prop.maxGridSize[i] << std::endl;
    return 0;
}