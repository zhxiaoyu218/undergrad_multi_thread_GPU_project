#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <malloc.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CHECK(call)																\
{																				\
	const cudaError_t error = call;												\
	if(error != cudaSuccess)													\
	{																			\
		printf("\nERROR:  %s : %d, ",__FILE__,__LINE__);						\
		printf("code:%d, reason: %s\n",error, cudaGetErrorString(error));		\
		exit(1);																\
	}																			\
}

//double			RotAngle;					// rotation angle

using namespace cv;


//void* (*RotateFunc)(void *arg);				// Function pointer to rotate the image (multi-threaded)

long 				Vpixels_Image,Hpixels_Image;
	

__global__  void rotate(long Vpixels_Image,long Hpixels_Image,double RotAngleGPU,unsigned char*	TheImage_GPU,unsigned char*	CopyImage_GPU,long ChunkSize)
{
	
    int row,col,h,v,c, hp3;
	double cc, ss, k1, k2;
	int NewRow,NewCol;
	double X, Y, newX, newY, ScaleFactor;
	double Diagonal, H, V;
	double CRA,SRA, CRAS, SRAS, SRAYS, CRAYS;
    struct Pixel pix;

	           		     // My thread number (ID) is stored here
	long sp=(blockIdx.x * blockDim.x +threadIdx.x)*threadDim;
	
    //tn = *((int *) tid);           // Calculate my Thread ID
   // tn *= Vpixels_Image;
   
	
			H=(double)Hpixels_Image;
			V=(double)Vpixels_Image;
			Diagonal=sqrt(H*H+V*V);
			ScaleFactor=(Hpixels_Image>Vpixels_Image) ? V/Diagonal : H/Diagonal;
			
			CRA=cos(RotAngleGPU);	CRAS=ScaleFactor*CRA;
			SRA=sin(RotAngleGPU);	SRAS=ScaleFactor*SRA;
			h=Hpixels_Image/2;   v=Vpixels_Image/2;	// integer div
			hp3=Hpixels_Image*3;
    
	
	for(row=sp; row<sp+ChunkSize;row++)
	{
			col=0;
			cc=0.00;
			ss=0.00;
		
			Y=(double)v-(double)row;
			SRAYS=SRAS*Y;    
			CRAYS=CRAS*Y;
			k1=CRAS*(double)h + SRAYS;
			k2=SRAS*(double)h - CRAYS;

			while(col<hp3)	
			{
				newX=cc-k1;
				newY=ss-k2;
				NewCol=((int) newX+h);
				NewRow=v-(int)newY; 
				if((NewCol>=0) && (NewRow>=0) && (NewCol<Hpixels_Image) && (NewRow<Vpixels_Image))
				{
					NewCol = NewCol + NewCol + NewCol;	
					
				CopyImage_GPU[NewRow*hp3+NewCol]   = TheImage_GPU[row*hp3+col];
				CopyImage_GPU[NewRow*hp3+NewCol+1] = TheImage_GPU[row*hp3+col+1];
				CopyImage_GPU[NewRow*hp3+NewCol+2] = TheImage_GPU[row*hp3+col+2];

					
					
				} 		
				col+=3;		
				cc += CRAS;
				ss += SRAS;
			}
    }		
}



int main(int argc, char** argv)
{
//****************    HOST   variable ******************

    int 				a,i;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	char				FuncName[50];
	
//*********************    GPU    variable    ************************
	int					ImageNum;
	double* 			RotDegree;


	unsigned char*		temp_image;
	unsigned char**		temp_copy;
	
	unsigned char*		TheImage_GPU;					
	unsigned char*		CopyImage_GPU;	
	long				NumOfBlock;
	long 				ChunkSize,BlockSize;
	
	
	


    if (argc！=4)
	{
		printf("\nUsage: %s inputBMP outputPNG NumberOfImages\n\n");
		printf("Nothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
    }
	
	//ImageNum = atoi(argv[3]);
	ImageNum =1;
	RotDegree = (double *)malloc(ImageNum * sizeof(double));	
	for(i=0;i<ImageNum;i++)
	{
		RotDegree[i]= (360 * i / ImageNum) * 3.141593 /180;
	}
//******************   image  ************************	

	mat image;
	image= imread(argv[1], CV_LOAD_IMAGE_COLOR);
		if(! image.data ) 
		{
			printf(Could not open or find the image.\n");
			exit(EXIT_FAILURE);
		}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);
		
		
		
	temp_image = (unsigned char *)malloc(image.rows*image.cols* sizeof(unsigned char));	
	
	CHECK(	cudaMalloc ((unsigned char **)&TheImage_GPU, image.rows*image.cols*sizeof(unsigned char))    );
	for(i=0; i< image.cols i++)
	{
		memcpy((unsigned char *) &temp_image[i*image.cols], (unsigned char *) TheImage[i], (size_t) image.cols);
    }
	CHECK(cudaMemcpy (TheImage_GPU, temp_image, image.rows*image.cols *sizeof(unsigned char), cudaMemcpyHostToDevice));	
	
	
//******************  Copy-image  ************************	
	CHECK(    cudaMalloc ((unsigned char **)&CopyImage_GPU, image.rows*image.cols*sizeof(unsigned char))    );			//create blank 1-D copy image GPU	
	
	
	temp_copy = (unsigned char **)malloc(ImageNum*sizeof(unsigned char*));
	for(i=0; i<ImageNum; i++)
	{
        temp_copy[i] = (unsigned char *)malloc(image.rows*image.cols * sizeof(unsigned char));
    }
	
	
	
	CopyImage = (unsigned char **)malloc(((image.rows)) * sizeof(unsigned char*));		//create blank 2-D copy image 
    for(i=0; i<image.rows; i++)
	{
        CopyImage[i] = (unsigned char *)malloc((image.cols) * sizeof(unsigned char));
    }

	Vpixels_Image  = image.rows;
	Hpixels_Image  = image.cols;
	
	BlockSize = 32;
	ChunkSize = 16;
	NumBlock = (image.rows*image.cols)/(BlockSize*ChunkSize);
	
	
	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	for(i=0;i<ImageNum;i++)	
	{	
		//<<<numBlocks, threadsPerBlock>>>
		rotate<<< NumOfBlock,BlockSize >>> (image.rows,Hpixels_Image,RotDegree[i],TheImage_GPU,CopyImage_GPU,ChunkSize);	/////////GPU function
		 CHECK(cudaMemcpy (temp_copy[i], CopyImage_GPU, image.rows*image.cols  *sizeof(unsigned char), cudaMemcpyDeviceToHost));	
	}
	CHECK(	cudaDeviceSynchronize()  );
	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;

	
    //merge with header and write to file
	Mat result = Mat(M, N, CV_8UC1, CPU_OutputArray);
	show_image(result, "output image");
    //WriteBMP(CopyImage, argv[2]);
	
 
 	// free() the allocated area for the images
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); free(CopyImage[i]); }
	free(TheImage);   free(CopyImage);   
   
    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);
	if(NumThreads>=1) printf("(%10.4f  Thread-ms)  ",TimeElapsed*(double)NumThreads);
    printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
	
	if(Function==9)
	{
		printf("\n\n============================================\n");
		for(i=0; i<NumThreads; i++)
		{	
			printf("\ntid= %2li processed %4d rows \n",i,ThParam[i]*2);
		}
		printf("\n\n============================================\n");
	}
	
    

//*******************************************		
		case 'D':
					free(InputArrayD);
					free(SortedArrayD);
					cudaFree(InputArrayG_D);
					cudaFree(SortedArrayG_D);
		break;
	}

    return (EXIT_SUCCESS);
}

