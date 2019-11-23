NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/local/include/opencv4

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

# or if using MacPorts

#OPENCV_LIBPATH=/opt/local/lib
#OPENCV_INCLUDEPATH=/opt/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda-10.1/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

NVCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -m64 -arch=sm_75

GCC_OPTS=-O3 -Wall -Wextra -m64 -Wmaybe-uninitialized -Wunused-parameter

#student: main.o Makefile
#	$(NVCC) -o HW1 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main: main.o calculations.o Makefile
	$(NVCC) -o main main.o calculations.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

main.o: main.cpp utils.h process.cpp
	g++ -c -g main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH) -I .

calculations.o: calculations.cu utils.h
	nvcc -c calculations.cu $(NVCC_OPTS)

clean:
	rm -f *.o main ./Output/*.JPG log.txt ./LightFrames/Stacked/*.JPG log.txt
