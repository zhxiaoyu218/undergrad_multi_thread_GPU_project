#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <malloc.h>

#include "ImageStuff.h"

#define REPS 	     1
#define MAXTHREADS   128
#define N 			 32 

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...

pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes


unsigned char**	TheImage;					// This is the main image
unsigned char**	CopyImage;					// This is the copy image

struct ImgProp 	ip;

		




__global__ void shrunk(int xshrink, int yshrink,unsigned char **ImageGPU ,unsigned char **CopyImage_GPU)
{          		     // My thread number (ID) is stored here
    int row,col,hp3;

	int NewRow,NewCol;
	int xskip,yskip;
	int xbound,ybound;
	
	
	//yskip = yshrink -1;
	
	ybound = ip.Vpixels/yshrink;
	xbound = ip.Hpixels*3/xshrink;

	hp3=ip.Hpixels*3;
	

	NewRow=0;
	row=0;

	while(NewRow < ybound)
	{
        col=0;
		NewCol=0;
		
        while(NewCol <xbound )
		{		
			CopyImage_GPU[NewRow][NewCol]   = ImageGPU[row][col];
			CopyImage_GPU[NewRow][NewCol+1] = ImageGPU[row][col+1];
			CopyImage_GPU[NewRow][NewCol+2] = ImageGPU[row][col+2];
			NewCol +=3; 			
            col+=xshrink + xshrink + xshrink ;
        }
		
		row+=yshrink;
		NewRow++;
    }
}


int main(int argc, char** argv)
{
    int 				a,i,ThErr;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	char				FuncName[50];
	int               xshrink, yshrink;

	// GPU variable	
	int threadsPerBlock = 16;  
    int numBlocks = N/threadsPerBlock;
	unsigned char**	ImageGPU;					
	unsigned char**	CopyImage_GPU;			
	
	
	NumThreads = 8;
	xshrink = atoi(argv[3]);
	yshrink = atoi(argv[4]);
	
	

	if(argc != 5)
	{
		printf("\nUsage:  inputBMP outputBMP xshrink yshrink \n\n");
		printf("Nothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
		
	}
	TheImage = ReadBMP(argv[1]);
	
	printf("\nVpixels: %i   Hpixels: %i \n",ip.Vpixels,ip.Hbytes);
	printf("yshrink: %i   xshrink: %i \n",(ip.Vpixels/yshrink ),((ip.Hbytes/xshrink)));
	
	//**********************
	CopyImage= (unsigned char **)malloc(((ip.Vpixels/yshrink )) * sizeof(unsigned char*));
    for(i=0; i<(ip.Vpixels/yshrink ); i++)
	{
        CopyImage[i] = (unsigned char *)malloc((ip.Hbytes/xshrink) * sizeof(unsigned char));
    }
	
	
	
	cudaMallloc ((unsigned char **)&CopyImage_GPU,(ip.Vpixels/yshrink ) * sizeof(unsigned char*));
	cudaMallloc ((unsigned char **)&ImageGPU,(ip.Vpixels ) * sizeof(unsigned char*));
	
	for(i=0; i<(ip.Vpixels/yshrink ); i++)
	{
       cudaMallloc ((unsigned char *)&CopyImage_GPU[i],(ip.Hbytes/yshrink ) * sizeof(unsigned char));
	   cudaMallloc ((unsigned char *)&ImageGPU[i],ip.Hbytes * sizeof(unsigned char));
    }
	
	cudaMemcpy2D(ImageGPU,TheImage, cudaMemcpyHostToDevice);
	//cudaMemcpy(CopyImageGPU,CopyImage, cudaMemcpyHostToDevice);

	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);

	for(a=0; a<REPS; a++)
	{
			 shrunk<<<numBlocks, threadsPerBlock>>>(xshrink,yshrink，ImageGPU ,CopyImage_GPU);
	}

    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;
    //merge with header and write to file
	
	ip.Hbytes  /= xshrink;
	ip.Vpixels /= yshrink;
	ip.Hpixels /= xshrink;
	
	cudaMemcpy2D(CopyImage,CopyImageGPU, cudaMemcpyDeviceToHost)
    WriteBMP(CopyImage, argv[2]);

 	// free() the allocated area for the images
	for(i = 0; i < ip.Vpixels; i++) 
	{ 
		free(TheImage[i]); 
		free(CopyImage[i]);
	}
	free(TheImage);   
	free(CopyImage);   
   
   
    cudaFree(ImageGPU);  
    cudaFree(CopyImageGPU);
	cudaFree(xshrink);
	cudaFree(yshrink);
   
    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);
	if(NumThreads>=1) printf("(%10.4f  Thread-ms)  ",TimeElapsed*(double)NumThreads);
    printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));

    return (EXIT_SUCCESS);
}
