#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utils.h"
#include <typeinfo>

cv::Mat inputImage;
cv::Mat outputImage;

uchar4 *d_frame__;
uchar4 *h_frame__;
int NUMBER_OF_IMAGES = 0;

size_t numRows() { return inputImage.rows; }
size_t numCols() { return inputImage.cols; }
int getNumberOfImages() { return NUMBER_OF_IMAGES; }

void processLightFrames(uchar4* d_lightFrames, uchar4* d_outputLightFrame, int width, int height, int numberOfImages);
void subtractFrames(uchar4* d_outputLightFrame, uchar4* d_outputDarkFrame, uchar4* d_FinalImage, int width, int height);

using namespace cv;

bool has_suffix(const std::string &str, const std::string &suffix)
{
	return str.size() >= suffix.size() &&
	str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void preProcessFrames(uchar4** h_frames, uchar4** d_frames, uchar4** d_outputFrame, std::string light_frames_folder, std::string frame_type) 
{
	checkCudaErrors(cudaFree(0));
	std::cout << "Pre Processing " << frame_type << " Frames: " << std::endl;
	
	int image_number = 0;
	struct dirent *entry;
	DIR *dir = opendir(light_frames_folder.c_str());

	if (dir == NULL) {
		return;
	}
	while ((entry = readdir(dir)) != NULL) 
	{
		if(has_suffix(light_frames_folder + entry->d_name, ".jpg")) 
		{
			image_number++;
			inputImage = cv::imread(light_frames_folder + entry->d_name, IMREAD_COLOR);
		}
	}
	NUMBER_OF_IMAGES = image_number;
	image_number = 0;
	size_t numPixels = numRows() * numCols();
	size_t numberOfImages = getNumberOfImages();
	std::cout << numPixels << " " << numberOfImages << std::endl;
	*h_frames = (uchar4*) malloc(sizeof(uchar4) * numPixels * numberOfImages);
	dir = opendir(light_frames_folder.c_str());
	if(h_frames != NULL) 
	{
		while ((entry = readdir(dir)) != NULL) 
		{
			if(has_suffix(light_frames_folder + entry->d_name, ".jpg")) 
			{
				std::cout << "Processing Image: " << entry->d_name << std::endl;
				cv::Mat image;
				image = cv::imread(light_frames_folder + entry->d_name, IMREAD_COLOR);
				cv::cvtColor(image, inputImage, COLOR_BGR2RGBA);
				std::cout << "Pixel Value: " << (int)(*(uchar4*)inputImage.ptr<unsigned char>(0)).x << " " << std::endl;
				memcpy (*h_frames + (numPixels * image_number), (uchar4*)inputImage.ptr<unsigned char>(0), sizeof(uchar4) * numPixels);
				image_number++; 
			}
		}
		closedir(dir);

		//Allocate memory for the frames on device
		checkCudaErrors(cudaMalloc((void**)d_frames, sizeof(uchar4) * numPixels * numberOfImages));
		//Allocate memory for the output frame
		checkCudaErrors(cudaMalloc((void**)d_outputFrame, sizeof(uchar4) * numPixels));
		//copy all the light frames data to the GPU
		checkCudaErrors(cudaMemcpy((void*)*d_frames, (void*)*h_frames, sizeof(uchar4) * numPixels * numberOfImages, cudaMemcpyHostToDevice));
		//Set all the pixel values to 0 for the output frame
		checkCudaErrors(cudaMemset((void *)*d_outputFrame, 0, sizeof(uchar4) * numPixels));
		d_frame__ = *d_frames;

		outputImage.create(numRows(), numCols(), CV_8UC4);
	}
}

void allocateMemoryForFinalImage(uchar4** d_outputFrame) 
{
	size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMalloc((void**)d_outputFrame, sizeof(uchar4) * numPixels));
	//Initialize the data to 0
	checkCudaErrors(cudaMemset((void *)*d_outputFrame, 0, sizeof(uchar4) * numPixels));
	cudaDeviceSynchronize();
}

void postProcess(uchar4* data_ptr, std::string output_path)
{
	std::cout << "Post Processing Image..." << std::endl;
	size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMemcpy((uchar4*)outputImage.ptr<unsigned char>(0), data_ptr, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
	cv::Mat imageOutputBGR;
	std::cout << "Pixel Value: " << (int)outputImage.at<uchar4>(0, 0).x << " " << std::endl;
	cv::cvtColor(outputImage, imageOutputBGR, COLOR_RGBA2BGR);	
	cv::imwrite(output_path.c_str(), imageOutputBGR);
}

void stackFrames(uchar4* d_lightFrames, uchar4* d_outputLightFrame){
	int height = numRows();
	int width = numCols();
	int numberOfImages = getNumberOfImages();
	processLightFrames(d_lightFrames, d_outputLightFrame, width, height, numberOfImages);
}

void subtractFramesCall(uchar4* d_outputLightFrame, uchar4* d_outputDarkFrame, uchar4* d_finalOutputFrame)
{
	int height = numRows();
	int width = numCols();
	subtractFrames(d_outputLightFrame, d_outputDarkFrame, d_finalOutputFrame, width, height);
}

void cleanup()
{
	cudaFree(d_frame__);
}