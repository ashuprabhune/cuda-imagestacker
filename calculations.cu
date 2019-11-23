#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utils.h"

__global__
void calculateAverageLightFrames(const uchar4* d_lightFrames, uchar4* d_outputFrame, int width, int height)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  	if(index_x > width || index_y > height)
  		return;
  	// map the two 2D indices to a single linear, 1D index
  	int grid_width = gridDim.x * blockDim.x;
  	int index = index_y * grid_width + index_x;
  	uchar4 newValue;
  	for(int i = 0; i < 9; i++)
  	{
  		uchar4 currentPixelValue = d_lightFrames[index + ((width * height) * i) + 1];
  		uchar4 oldPixelValue = d_outputFrame[index];
  		
  		newValue.x += currentPixelValue.x;
  		newValue.y += currentPixelValue.y;
  		newValue.z += currentPixelValue.z;
  	}
  	newValue.x = newValue.x/9;
  	newValue.y = newValue.y/9;
  	newValue.z = newValue.z/9;
  	d_outputFrame[index] = newValue;
}

// __global__
// void calculateAverageLightFrames(uchar4* d_outputLightFrame, int numberOfImages)
// {
// 	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
//   	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

//   	// map the two 2D indices to a single linear, 1D index
//   	int grid_width = gridDim.x * blockDim.x;
//   	int index = index_y * grid_width + index_x;
//   	uchar4 pixelValue = d_outputLightFrame[index];
//   	uchar4 newValue;
//   	newValue.x = pixelValue.x/numberOfImages;
//   	newValue.y = pixelValue.y/numberOfImages;
//   	newValue.z = pixelValue.z/numberOfImages;
//   	d_outputLightFrame[index] = newValue;
// }

void averageOfLightFrames(uchar4* d_lightFrames, uchar4* d_outputLightFrame, int width, int height)
{
	std::cout << "Calculating sum of the Light Frames" << std::endl;
	const int thread = 16;
	const dim3 blockSize(thread, thread);
	const dim3 gridSize(ceil(height/(float)thread), ceil(width/(float)thread));
	calculateAverageLightFrames<<<gridSize, blockSize>>>(d_lightFrames, d_outputLightFrame, width, height);
	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
}