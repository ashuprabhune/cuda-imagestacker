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

#define WIDTH   			6000
#define HEIGHT  			4000
#define NUMBER_OF_IMAGES  	9

cv::Mat inputImage;
cv::Mat outputImage;

uchar4 *d_frame__;
uchar4 *h_frame__;

using namespace cv;

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void preProcessFrames(uchar4** h_frames, uchar4** d_frames, uchar4** d_outputFrame, uchar4** h_outputFinalFrame, std::string light_frames_folder, std::string frame_type) 
{
	checkCudaErrors(cudaFree(0));
	std::cout << "Pre Processing " << frame_type << " Frames: " << std::endl;
	size_t numPixels = WIDTH * HEIGHT;
	size_t numberOfImages = NUMBER_OF_IMAGES;
	int image_number = 0;
	*h_frames = (uchar4*) malloc(sizeof(uchar4) * numPixels * numberOfImages);
	//memset(*h_frames, 0, sizeof(uchar4) * numPixels * numberOfImages);

	if(h_frames != NULL) 
	{
		struct dirent *entry;
	   	DIR *dir = opendir(light_frames_folder.c_str());
	   
	   	if (dir == NULL) {
	      	return;
	   	}
	   	while ((entry = readdir(dir)) != NULL) 
		{
			if(has_suffix(light_frames_folder + entry->d_name, ".JPG")) 
			{
				std::cout << "Processing Image: " << entry->d_name << std::endl;
				cv::Mat image;
	  			image = cv::imread(light_frames_folder + entry->d_name, IMREAD_COLOR);
	  			std::cout << "Pixel Value: " << (int)image.at<uchar4>(0, 0).x << " " << std::endl;
	  			cv::cvtColor(image, inputImage, COLOR_RGB2BGRA);
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
	  	d_frame__ = *d_outputFrame;

	  	outputImage.create(HEIGHT, WIDTH, CV_8UC4);
	  	*h_outputFinalFrame  = outputImage.ptr<uchar4>(0);

	}
}

void allocateMemoryForFinalImage(uchar4** d_outputFrame) 
{
	size_t numPixels = WIDTH * HEIGHT;
	size_t numberOfImages = NUMBER_OF_IMAGES;
	checkCudaErrors(cudaMalloc((void**)d_outputFrame, sizeof(uchar4) * numPixels));
	//Initialize the data to 0
	checkCudaErrors(cudaMemset((void *)*d_outputFrame, 0, sizeof(uchar4) * numPixels));

	uchar4* temp;
	temp = (uchar4*) malloc(sizeof(uchar4) * numPixels);
	checkCudaErrors(cudaMemcpy(temp, d_outputFrame, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

void postProcess(uchar4* data_ptr, std::string output_path)
{
	std::cout << "Post Processing Image: " << std::endl;
	size_t numPixels = WIDTH * HEIGHT;
	checkCudaErrors(cudaMemcpy(data_ptr, d_frame__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
	cv::Mat output(HEIGHT, WIDTH, CV_8UC4, (void*)data_ptr);
	cv::cvtColor(output, outputImage,COLOR_BGR2RGBA);	
	cv::imwrite(output_path.c_str(), outputImage);
	//cv::imshow("result", outputImage);
	std::cout << (int)outputImage.at<uchar4>(0, 0).x <<std::endl;
	//cv::waitKey(0);
}

void checkData(uchar4* data_ptr, std::string output_path) 
{
	std::cout << "Checking Image: " << std::endl;
	size_t numPixels = WIDTH * HEIGHT;
	checkCudaErrors(cudaMemcpy(data_ptr, d_frame__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	cv::Mat output(HEIGHT, WIDTH, CV_8UC4, (void*)data_ptr);
	cv::Mat mahitnahi;
	cv::cvtColor(output, mahitnahi,COLOR_BGR2RGBA);	
	cv::imwrite(output_path.c_str(), outputImage);

}

void cleanup()
{
  //cleanup
  cudaFree(d_frame__);
}