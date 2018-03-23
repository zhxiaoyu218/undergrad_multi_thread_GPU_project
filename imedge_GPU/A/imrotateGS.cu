#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;




cudaError_t launch_rotateG(Mat image, unsigned char *CPU_OutputArray, float* Runtimes, int row_image, int col_image, double RotAngle);


__global__ void rotateG(unsigned char *GPU_i, unsigned char *GPU_o, int row_image, int col_image, double  RotAngle)
{

	int row,col,h,v,Hbytes_image;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double  CRAS, SRAS;
		

	int sp = blockIdx.x * blockDim.x + threadIdx.x;   //the ID of the row
	
//******************************   shared memory **********************
	extern __shared__ unsigned char Buffer[];
	
	
	H=(double)col_image;			V=(double)row_image;
	
	Hbytes_image =col_image*3;	
	Diagonal=sqrt(H*H+V*V);
	
	ScaleFactor=(col_image>row_image) ? row_image/Diagonal : col_image/Diagonal;
			
	CRAS = cos(RotAngle)*ScaleFactor;
	SRAS = sin(RotAngle)*ScaleFactor;
			
	h=col_image/2;   v=row_image/2;	
	row = sp;		 col=0;
	cc=0.00;		 ss=0.00;
	
	
	
	Y=(double)(v-row);
	k1=CRAS*(double)h + SRAS*Y;
	k2=SRAS*(double)h - CRAS*Y;
	
	memcpy((void *)&Buffer[threadIdx.x*Hbytes_image], (void *)&GPU_i[row*Hbytes_image], (size_t) Hbytes_image);
    while(col<Hbytes_image)
	{

		newX=cc-k1;
		newY=ss-k2;
		NewCol=((int) newX+h);
		NewRow=v-(int)newY;     
			
		if((NewCol>=0) && (NewRow>=0) && (NewCol<col_image) && (NewRow<row_image))
		{
				NewCol =NewCol+NewCol+NewCol;
				
				GPU_o[NewRow*Hbytes_image + NewCol]	  = Buffer[threadIdx.x*Hbytes_image + col];
				GPU_o[NewRow*Hbytes_image + NewCol+1] = Buffer[threadIdx.x*Hbytes_image + col+1];
				GPU_o[NewRow*Hbytes_image + NewCol+2] = Buffer[threadIdx.x*Hbytes_image + col + 2];

        }	
        col+=3;
		cc += CRAS;
		ss += SRAS;
    }
  
}

int main(int argc, char** argv)
{
	int 				 i;
	float GPURuntimes[4];		// run times of the GPU code
	float TotalTime = 0.0;
	unsigned char FILL = 255;
	cudaError_t cudaStatus;

//************     image variable     *************
	int row_image,col_image;
	unsigned char *CPU_OutputArray;

////************     arg parameter      ************* 	
	int N;
	double *RotAngle;
	char OutputImageName[50];
	 
	 if(argc != 4)
	{
		printf("\nUsage: imrotateG [input] [output] N\n");
		printf("\nExample: ./imrotateG Astronaut.bmp AROT.png 9\n\n");
		exit(EXIT_FAILURE);
	}
	
	
	N = atoi(argv[3]);
	if((N<=0) || (N>30))
	{
       		printf("\n   The N must to be between 0 to 30 \n");
			printf("\n\nNothing executed ... Exiting ...\n\n");
        	exit(EXIT_FAILURE);
	}
	printf("N = %d\n", N);

	
	RotAngle = (double*)malloc(N*sizeof(double));
	for(i = 0; i<N;i++)
	{
		RotAngle[i] = 3.1415926*2/N*i;   // calculate and transform angle to radian
	}
	
//***********   openCV read image   ************
	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(! image.data ) 
	{
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);
	
	row_image = image.rows;
	col_image = image.cols;        
	
	
	
//***********    generate output image   **********
	CPU_OutputArray = (unsigned char*)malloc(row_image*col_image*3*sizeof(unsigned char));
	if (CPU_OutputArray == NULL) 
	{
		fprintf(stderr, "OOPS. Can't create CPU_OutputArray using malloc() ...\n");
		exit(EXIT_FAILURE);
	}

// ******* set up the string of the name *******
	char *name = argv[2];
	int  n = strlen(name);
	memcpy(OutputImageName, name, n-4);	

	for(i = 0; i<N; i++)
	{
//***************    get each one rotation of image  ******************		
			memset(CPU_OutputArray,FILL, row_image*col_image*3*sizeof(unsigned char)); 
			cudaStatus = launch_rotateG(image, CPU_OutputArray, GPURuntimes, row_image, col_image, RotAngle[i]);
			if (cudaStatus != cudaSuccess) 
			{
						fprintf(stderr, "\n launching failed!\n ");
						free(CPU_OutputArray);
						free(RotAngle);
						exit(EXIT_FAILURE);
			}

			TotalTime += GPURuntimes[2];	
			
//***************    store each one image  ******************
			Mat result = Mat(row_image, col_image, CV_8UC3, CPU_OutputArray);
			sprintf(&OutputImageName[n-4], "%d", i+1);	
			strcat(OutputImageName, ".png");				///

			if (!imwrite(OutputImageName, result)) 
			{
				fprintf(stderr, "couldn't write output to disk!\n");
				free(CPU_OutputArray);
				free(RotAngle);
				exit(EXIT_FAILURE);
			}	
	}
    
   
	printf("\nThe total Execution time = %7.4f ms ... \n",  TotalTime);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceReset failed!\n");
		free(CPU_OutputArray);
		free(RotAngle);
		exit(EXIT_FAILURE);
	}
 
//*********   free all memory of ary   ***************
 	free(RotAngle);
	free(CPU_OutputArray);    
	return (EXIT_SUCCESS);
}
 

// Helper function for using CUDA to add vectors in parallel.
cudaError_t launch_rotateG(Mat image, unsigned char *CPU_OutputArray, float* Runtimes, int row_image, int col_image, double RotAngle)
{
	cudaEvent_t time1, time2, time3, time4;
	int TotalGPUSize;

	unsigned char *GPU_idata;
	unsigned char *GPU_odata;
	
	int threadsPerBlock;
	int numBlocks;

	
	
   	 // Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
   	 if (cudaStatus != cudaSuccess) {
       		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        	goto Error;
   	 }

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);
    	
    	// Allocate GPU buffer for inputs and outputs 
    TotalGPUSize = row_image*col_image*3*sizeof(unsigned char);
	
   	 cudaStatus = cudaMalloc((void**)&GPU_idata, TotalGPUSize);
    	if (cudaStatus != cudaSuccess) {
        	fprintf(stderr, "cudaMalloc failed!");
        	goto Error;
    	}

    	cudaStatus = cudaMalloc((void**)&GPU_odata, TotalGPUSize);
    	if (cudaStatus != cudaSuccess) {
        	fprintf(stderr, "cudaMalloc failed!");
        	goto Error;
    	} 

    	// Copy input vectors from host memory to GPU buffers.
    	cudaStatus = cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
    	if (cudaStatus != cudaSuccess) {
       		 fprintf(stderr, "cudaMemcpy failed!");
        	goto Error;
   	 }
   	 
   	 cudaStatus = cudaMemcpy(GPU_odata, CPU_OutputArray, TotalGPUSize, cudaMemcpyHostToDevice);
    	if (cudaStatus != cudaSuccess) {
       		 fprintf(stderr, "cudaMemcpy failed!");
        	goto Error;
   	 }
 	
	cudaEventRecord(time2, 0);
	
	threadsPerBlock = 2;				//********************************************************
	numBlocks = row_image/threadsPerBlock;
	
	
	
	rotateG<<<numBlocks, threadsPerBlock, threadsPerBlock*3*col_image*sizeof(unsigned char)>>>(GPU_idata, GPU_odata, row_image, col_image, RotAngle);
		
		
	// Check for errors immediately after kernel launch.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaEventRecord(time3, 0);
    	
    	// Copy output (results) from GPU buffer to host (CPU) memory.
    	cudaStatus = cudaMemcpy(CPU_OutputArray, GPU_odata, TotalGPUSize, cudaMemcpyDeviceToHost); 
    	if (cudaStatus != cudaSuccess) {
      		  fprintf(stderr, "cudaMemcpy failed!");
       		 goto Error;
  	 }


	cudaEventRecord(time4, 0);
	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);


	Runtimes[0] = totalTime;
	Runtimes[1] = tfrCPUtoGPU;
	Runtimes[2] = kernelExecutionTime;
	Runtimes[3] = tfrGPUtoCPU;
	
	Error:
 	cudaFree(GPU_idata);
   	cudaFree(GPU_odata);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
    
    return cudaStatus;
}


