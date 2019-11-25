#include <iostream>
#include "process.cpp"

int main()
{
	std::string light_frames_folder = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/LightFrames/";
	std::string stacked_light_frame_path = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/LightFrames/Stacked/StackedLightFrame.JPG";
	std::string stacked_dark_frame_path = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/DarkFrames/Stacked/StackedDarkFrame.JPG";
  	std::string dark_frames_folder = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/DarkFrames/";
  	std::string output_file = "/home/abhishek/Desktop/Abhishek/UC/CS6068-PC/Project/ImageProcessing/Output/Output.JPG";

	uchar4 *d_lightFrames, *h_lightFrames, *d_outputLightFrame;
	uchar4 *d_darkFrames, *h_darkFrames, *d_outputDarkFrame;
	uchar4 *d_outputFrame;

	//Read all light frames and copy them to device memory
  	preProcessFrames(&h_lightFrames, &d_lightFrames, &d_outputLightFrame, light_frames_folder, "light");

  	//Calculate average of light frames
  	stackFrames(d_lightFrames, d_outputLightFrame);

  	//Post process the stacked Light Frame
  	postProcess(d_outputLightFrame, stacked_light_frame_path);

  	//Clear all the Light Frames Data
  	cleanup();

	std::cout << "############################################################################" << std::endl;
  	//Read all dark frames and copy them to device memory
  	preProcessFrames(&h_darkFrames, &d_darkFrames, &d_outputDarkFrame, dark_frames_folder, "dark");

  	//Calculate average of Dark frames
  	stackFrames(d_darkFrames, d_outputDarkFrame);

  	//Post process the stacked Dark Frame
  	postProcess(d_outputDarkFrame, stacked_dark_frame_path);

  	//Clear all the Dark Frames Data
  	cleanup();

	std::cout << "############################################################################" << std::endl;
  	//Allocate memory for final image
  	allocateMemoryForFinalImage(&d_outputFrame);

  	//Subtract Dark Frames from Light Frames
  	subtractFramesCall(d_outputLightFrame, d_outputDarkFrame, d_outputFrame);

  	//Post process the stacked Image
  	postProcess(d_outputFrame, output_file);

	cleanup();

}
