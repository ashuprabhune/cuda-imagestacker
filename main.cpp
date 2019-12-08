#include <iostream>
#include "process.cpp"
#include "timer.h"

int main()
{
	std::string light_frames_folder = "./Light/";
    std::string dark_frames_folder = "./Dark/";
    std::string bias_frames_folder = "./Bias/";
	std::string stacked_light_frame_path = "./Light/Stacked/master-light.JPG";
	std::string stacked_dark_frame_path = "./Dark/Stacked/master-dark.JPG";
    std::string stacked_bias_frame_path = "./Bias/Stacked/master-bias.JPG";
    std::string stacked_bias_subtracted_frame_path = "./Dark/bias-subtracted/bias-subtracted.JPG";
  	std::string output_file = "./Output/Output.JPG";

	uchar4 *d_lightFrames, *h_lightFrames, *d_outputLightFrame, *h_outputLightFrame;
	uchar4 *d_darkFrames, *h_darkFrames, *d_outputDarkFrame, *h_outputDarkFrame;
    uchar4 *d_biasFrames, *h_biasFrames, *d_outputBiasFrames, *h_outputBiasFrames;
	uchar4 *d_outputFrame;
    
    GpuTimer timer;

    std::cout << "##################################Processing Light Frames##########################################" << std::endl;
    //Read all light frames and copy them to device memory
  	preProcessFrames(&h_lightFrames, &d_lightFrames, &d_outputLightFrame, &h_outputLightFrame, light_frames_folder);

    //Calculate average of light frames
    timer.Start();
  	stackFrames(d_lightFrames, d_outputLightFrame);
    timer.Stop();

  	//Post process the stacked Light Frame
  	postProcess(d_outputLightFrame, stacked_light_frame_path);

    int err = printf("Time for stacking light frames (Parallel): %f msecs.\n", timer.Elapsed());

    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
    stackFramesSeqCall(h_lightFrames, h_outputLightFrame);

  	//Clear all the Light Frames Data
  	cleanup();

    std::cout << "##################################Processing Dark Frames##########################################" << std::endl;
  	//Read all dark frames and copy them to device memory
  	preProcessFrames(&h_darkFrames, &d_darkFrames, &d_outputDarkFrame, &h_outputDarkFrame, dark_frames_folder);

  	//Calculate average of Dark frames
    timer.Start();
  	stackFrames(d_darkFrames, d_outputDarkFrame);
    timer.Stop();

  	//Post process the stacked Dark Frame
  	postProcess(d_outputDarkFrame, stacked_dark_frame_path);

    err = printf("Time for stacking dark frames (Parallel): %f msecs.\n", timer.Elapsed());
    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
    stackFramesSeqCall(h_darkFrames, h_outputDarkFrame);

  	//Clear all the Dark Frames Data
  	cleanup();

    std::cout << "##################################Processing Bias Frames##########################################" << std::endl;
    //Read all Bias frames and copy them to device memory
    preProcessFrames(&h_biasFrames, &d_biasFrames, &d_outputBiasFrames, &h_outputBiasFrames, dark_frames_folder);

    //Calculate average of Bias frames
    timer.Start();
    stackFrames(d_biasFrames, d_outputBiasFrames);
    timer.Stop();

    //Post process the stacked Dark Frame
    postProcess(d_outputBiasFrames, stacked_bias_frame_path);

    err = printf("Time for stacking Bias frames (Parallel): %f msecs.\n", timer.Elapsed());
    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
    stackFramesSeqCall(h_biasFrames, h_outputBiasFrames);
    //Clear all the Bias Frames Data
    cleanup();

    //Allocate memory for final image
    allocateMemoryForFinalImage(&d_outputFrame);

    std::cout << "##################################Subtracting Bias from Dark##########################################" << std::endl;
    //Subtract Dark Frames from Light Frames
    timer.Start();
    subtractFramesCall(d_outputDarkFrame, d_outputBiasFrames, d_outputDarkFrame);
    timer.Stop();

    //Post process the stacked Image
    postProcess(d_outputDarkFrame, stacked_bias_subtracted_frame_path);

    err = printf("Time for subtracting Bias frame from Dark frame: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
    subtractFramesSeqCall(h_outputDarkFrame, h_outputBiasFrames, h_outputDarkFrame);

	std::cout << "##################################Subtracting Dark from light##########################################" << std::endl;

  	//Subtract Dark Frames from Light Frames
    timer.Start();
  	subtractFramesCall(d_outputLightFrame, d_outputDarkFrame, d_outputFrame);
    timer.Stop();

  	//Post process the stacked Image
  	postProcess(d_outputFrame, output_file);

    err = printf("Time for subtracting dark frame from light frame: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }
	cleanup();
    subtractFramesSeqCall(h_outputLightFrame, h_outputDarkFrame, h_outputLightFrame);

    delete[] h_lightFrames;
    delete[] h_outputLightFrame;
    delete[] h_darkFrames;
    delete[] h_outputDarkFrame;
    delete[] h_biasFrames;
    delete[] h_outputBiasFrames;
}
