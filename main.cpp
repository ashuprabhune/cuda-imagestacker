#include <iostream>
#include "process.cpp"
#include "timer.h"

int main()
{
	std::string light_frames_folder = "./LightFrames/";
	std::string stacked_light_frame_path = "./LightFrames/Stacked/StackedLightFrame.JPG";
	std::string stacked_dark_frame_path = "./DarkFrames/Stacked/StackedDarkFrame.JPG";
  	std::string dark_frames_folder = "./DarkFrames/";
  	std::string output_file = "./Output/Output.JPG";

	uchar4 *d_lightFrames, *h_lightFrames, *d_outputLightFrame;
	uchar4 *d_darkFrames, *h_darkFrames, *d_outputDarkFrame;
	uchar4 *d_outputFrame;

	//Read all light frames and copy them to device memory
  	preProcessFrames(&h_lightFrames, &d_lightFrames, &d_outputLightFrame, light_frames_folder, "light");

  	//Calculate average of light frames
    GpuTimer timer;
    timer.Start();
  	stackFrames(d_lightFrames, d_outputLightFrame);
    timer.Stop();

  	//Post process the stacked Light Frame
  	postProcess(d_outputLightFrame, stacked_light_frame_path);

    int err = printf("Time for stacking light frames: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

  	//Clear all the Light Frames Data
  	cleanup();

    std::cout << "############################################################################" << std::endl;
  	//Read all dark frames and copy them to device memory
  	preProcessFrames(&h_darkFrames, &d_darkFrames, &d_outputDarkFrame, dark_frames_folder, "dark");

  	//Calculate average of Dark frames
    timer.Start();
  	stackFrames(d_darkFrames, d_outputDarkFrame);
    timer.Stop();

  	//Post process the stacked Dark Frame
  	postProcess(d_outputDarkFrame, stacked_dark_frame_path);

    err = printf("Time for stacking dark frames: %f msecs.\n", timer.Elapsed());
    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

  	//Clear all the Dark Frames Data
  	cleanup();

	std::cout << "############################################################################" << std::endl;
  	//Allocate memory for final image
  	allocateMemoryForFinalImage(&d_outputFrame);

  	//Subtract Dark Frames from Light Frames
    timer.Start();
  	subtractFramesCall(d_outputLightFrame, d_outputDarkFrame, d_outputFrame);
    timer.Stop();

  	//Post process the stacked Image
  	postProcess(d_outputFrame, output_file);

    err = printf("Time for subtracting dark frames from light frames: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

	cleanup();

}
