#include <iostream>
#include "process.cpp"
#define WIDTH   			6000
#define HEIGHT  			4000
#define NUMBER_OF_IMAGES  	9

void calculationsSumOfLightFrames(uchar4* d_lightFrames, uchar4* d_outputLightFrame, int width, int height);
void calculateAverageOfLightFrames(uchar4* d_outputLightFrame, int numberOfImages, int width, int height);

int main()
{
  
	std::string light_frames_folder = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/LightFrames/";
	std::string stacked_light_frame_path = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/LightFrames/Stacked/StackedLightFrames.JPG";
  	std::string dark_frames_folder = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/DarkFrames/";
  	std::string output_file = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/Output/Output.JPG";

	uchar4 *d_lightFrames, *h_lightFrames, *d_outputLightFrame, *h_outputLightFrame;
	uchar4 *d_darkFrames, *h_darkFrames, *d_outputDarkFrames;
	uchar4 *d_outputFrame, *h_outputFrame;

	//Read all light frames and copy them to device memory
  	preProcessFrames(&h_lightFrames, &d_lightFrames, &d_outputLightFrame, &h_outputLightFrame, light_frames_folder, "light");

  	//Calculate average of light frames
  	averageOfLightFrames(d_lightFrames, d_outputLightFrame, WIDTH, HEIGHT);

  	//Post process the stacked Light Frame
  	postProcess(h_outputLightFrame, stacked_light_frame_path);

  	//Read all dark frames and copy them to device memory
  	//preProcessFrames(&h_darkFrames, &d_darkFrames, &d_outputDarkFrames, dark_frames_folder, "dark");

  	//Allocate memory for final image
  	//allocateMemoryForFinalImage(&d_outputFrame);

  	//Calculate average of the frames
  	//calculateAverageOfLightFrames();

  	//Calculate median of the dark frames
  	//calculateMedianOfDarkFrames();

  	//Subtract Dark Frames from Light Frames
  	//subtractFrames();

  	//Store the final image
  	//postProcess();

	//checkData(h_outputLightFrame, output_file);

	cleanup();

}
