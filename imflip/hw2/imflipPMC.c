#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "ImageStuff.h"

#define REPS         101
#define MAXTHREADS   128
#define BUFFER_SIZE  48*1024

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes
void(*FlipFunc)(unsigned char** img);		// Function pointer to flip the image
void* (*MTFlipFunc)(void *arg);				// Function pointer to flip the image, multi-threaded version

unsigned char**	TheImage;					// This is the main image
struct ImgProp 	ip;

void FlipImageV(unsigned char** img)
{
   struct Pixel pix; //temp swap pixel
    int row, col;
    
    //vertical flip
    for(col=0; col<ip.Hbytes; col+=3)
    {
        row = 0;
        while(row<ip.Vpixels/2)
        {
            pix.B = img[row][col];
            pix.G = img[row][col+1];
            pix.R = img[row][col+2];

            img[row][col]   = img[ip.Vpixels-(row+1)][col];
            img[row][col+1] = img[ip.Vpixels-(row+1)][col+1];
            img[row][col+2] = img[ip.Vpixels-(row+1)][col+2];
            
            img[ip.Vpixels-(row+1)][col]   = pix.B;
            img[ip.Vpixels-(row+1)][col+1] = pix.G;
            img[ip.Vpixels-(row+1)][col+2] = pix.R;
            
            row++;
        }
    }
}


void FlipImageH(unsigned char** img)
{
    struct Pixel pix; //temp swap pixel
    int row, col;
    
    //horizontal flip
    for(row=0; row<ip.Vpixels; row++)
    {
        col = 0;
        while(col<(ip.Hpixels*3)/2)
        {
            pix.B = img[row][col];
            pix.G = img[row][col+1];
            pix.R = img[row][col+2];
            
            img[row][col]   = img[row][ip.Hpixels*3-(col+3)];
            img[row][col+1] = img[row][ip.Hpixels*3-(col+2)];
            img[row][col+2] = img[row][ip.Hpixels*3-(col+1)];
            
            img[row][ip.Hpixels*3-(col+3)] = pix.B;
            img[row][ip.Hpixels*3-(col+2)] = pix.G;
            img[row][ip.Hpixels*3-(col+1)] = pix.R;
            
            col+=3;
        }
    }
}


void *MTFlipHMC(void* tid)
{
	struct Pixel pix; //temp swap pixel
	int row, col;
	unsigned char Buffer[BUFFER_SIZE];	 // This is the buffer to use to get the entire row

	long ts = *((int *)tid);       	// My thread ID is stored here
	ts *= ip.Vpixels / NumThreads;			// start index
	long te = ts + ip.Vpixels / NumThreads - 1; 	// end index
	
	unsigned long int *p;
	unsigned long int a,b,c,d,e,f,temp1,temp2,temp3;
	int k;
	

			
	for (row = ts; row <= te; row++) 
	{
		memcpy((void *)Buffer, (void *)TheImage[row], (size_t)ip.Hbytes);	
		p = (unsigned long int *)Buffer;	
		col = 0;
		
		while (col<(ip.Hpixels  * 3 / 16))  
		{	 
		 //     |a|b|c|  ``````````|d|e|f|    
			a = p[col];				
			b = p[col+1];		
			c = p[col+2];
			
			d = p[ip.Hpixels *3/8-(col+1)];
			e = p[ip.Hpixels *3/8-(col+2)];
			f = p[ip.Hpixels *3/8-(col+3)];		


			p[ip.Hpixels *3/8-(col+1)] = (a << 40) + ((a>>8)& 0xFFFFFF0000) + (a>>56) + ((b <<56) >>48) ; 
        	p[ip.Hpixels *3/8-(col+2)] = ((a>>48)<<56) + ((b>>24)& 0xFFFFFF00) + ((b<<24)& 0xFFFFFF00000000) + ((c<<48)>>56) ;
        	p[ip.Hpixels *3/8-(col+3)] = (c >> 40) + ((c<<8)& 0xFFFFFF000000) + (c<<56) + ((b >>56)<<48);   
			
			p[col] = (d >> 40) + ((d<<8)& 0xFFFFFF000000) + (d<<56) + ((e >>56)<<48);
			p[col+1] = ((d>>8)& 0xFF) | ((e>>24)& 0xFFFFFF00) + ((e<<24)& 0xFFFFFF00000000) + ((e>>48)<<56);
			p[col+2] =(f << 40) + ((f>>8)& 0xFFFFFF0000) + (f>>56) + ((d<<8) & 0xFF00) ;
			
			col+=3;
		}

		memcpy((void *)TheImage[row], (void *)Buffer, (size_t)ip.Hbytes);
	}
	pthread_exit(NULL);
}

void *MTFlipHM(void* tid)
{
	struct Pixel pix; //temp swap pixel
	int row, col;
	unsigned char Buffer[BUFFER_SIZE];	 // This is the buffer to use to get the entire row

	long ts = *((int *)tid);       	// My thread ID is stored here
	ts *= ip.Vpixels / NumThreads;			// start index
	long te = ts + ip.Vpixels / NumThreads - 1; 	// end index

	for (row = ts; row <= te; row++) {
		memcpy((void *)Buffer, (void *)TheImage[row], (size_t)ip.Hbytes);
		col = 0;
		while (col<ip.Hpixels * 3 / 2) {
			pix.B = Buffer[col];
			pix.G = Buffer[col + 1];
			pix.R = Buffer[col + 2];

			Buffer[col] = Buffer[ip.Hpixels * 3 - (col + 3)];
			Buffer[col + 1] = Buffer[ip.Hpixels * 3 - (col + 2)];
			Buffer[col + 2] = Buffer[ip.Hpixels * 3 - (col + 1)];

			Buffer[ip.Hpixels * 3 - (col + 3)] = pix.B;
			Buffer[ip.Hpixels * 3 - (col + 2)] = pix.G;
			Buffer[ip.Hpixels * 3 - (col + 1)] = pix.R;

			col += 3;
		}
		memcpy((void *)TheImage[row], (void *)Buffer, (size_t)ip.Hbytes);
	}
	pthread_exit(NULL);
}

void *MTFlipVM(void* tid)
{
  int row, row2;
  unsigned char Buffer[BUFFER_SIZE];	 // This is the buffer to get the first row
  unsigned char Buffer2[BUFFER_SIZE];	 // This is the buffer to get the second row

  long ts = *((int *) tid);       	// My thread ID is stored here
  ts *= ip.Vpixels/NumThreads/2;				// start index
  long te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index
  
  for(row=ts; row<=te; row++){
    memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
    row2=ip.Vpixels-(row+1);   
    memcpy((void *) Buffer2, (void *) TheImage[row2], (size_t) ip.Hbytes);
    // swap row with row2
    memcpy((void *) TheImage[row], (void *) Buffer2, (size_t) ip.Hbytes);
    memcpy((void *) TheImage[row2], (void *) Buffer, (size_t) ip.Hbytes);
  }
  pthread_exit(NULL);
}

void *MTFlipVM2(void* tid)
{
  int row, row2;
  unsigned char Buffer[BUFFER_SIZE];	 // This is the buffer to get the first row
  //unsigned char Buffer2[BUFFER_SIZE];	 // This is the buffer to get the second row

  long ts = *((int *) tid);       	// My thread ID is stored here
  ts *= ip.Vpixels/NumThreads/2;				// start index
  long te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index
  
  for(row=ts; row<=te; row++){
    memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
    row2=ip.Vpixels-(row+1);   
    memcpy((void *) TheImage[row], (void *) TheImage[row2], (size_t) ip.Hbytes);
    // swap row with row2
   // memcpy((void *) TheImage[row], (void *) Buffer2, (size_t) ip.Hbytes);
    memcpy((void *) TheImage[row2], (void *) Buffer, (size_t) ip.Hbytes);
  }
  pthread_exit(NULL);
}


// 
void *MTFlipVM3(void* tid)
{
  int row, row2;
  unsigned char Buffer[BUFFER_SIZE];	 // This is the buffer to get the first row
  unsigned char Buffer2[BUFFER_SIZE];	 // This is the buffer to get the second row
  

  long ts = *((int *) tid);       	// My thread ID is stored here
  ts *= ip.Vpixels/NumThreads/2;				// start index
  long te = ts+(ip.Vpixels/NumThreads/2)-1; 	// end index
 // int index = ip.Hbytes+ip.Hbytes+16;
 // int index2 = ip.Hbytes+16;

  for(row=ts; row<=te; row += 2){
    	memcpy((void *) Buffer, (void *) TheImage[row], (size_t) ip.Hbytes);
		memcpy((void *) Buffer2, (void *) TheImage[row+1], (size_t) ip.Hbytes);
		row2=ip.Vpixels-(row+2); 
   	//memcpy((void *) Buffer2, (void *) TheImage[row2], (size_t) index);
    	// swap row with row2
        

		
    	memcpy((void *) TheImage[row+1], (void *)TheImage[row2+1], (size_t) ip.Hbytes);
    	memcpy((void *) TheImage[row],   (void *)TheImage[row2], (size_t) ip.Hbytes);
		memcpy((void *) TheImage[row2] , (void *)Buffer, (size_t) ip.Hbytes);
    	memcpy((void *) TheImage[row2+1], (void *)Buffer2, (size_t) ip.Hbytes);
    
  }
  pthread_exit(NULL);
}




int main(int argc, char** argv)
{
	char 				Flip;
	
    int 				a,i,ThErr;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	char				FlipType[50];
	
	
    switch (argc){
		case 3 : NumThreads=0; 				Flip = 'V';						break;
		case 4 : NumThreads=0;  			Flip = toupper(argv[4][0]);		break;
		case 5 : NumThreads=atoi(argv[3]);  Flip = toupper(argv[4][0]);		break;
		default: printf("\n\nUsage: imflipPm input output [v,h,w,i] [0,1-128]");
				 printf("\n\nUse 'V', 'H' for regular, and 'W', 'I' for the memory-friendly version of the program\n\n");
				 printf("\n\nNumThreads=0 for the serial version, and 1-128 for the Pthreads version\n\n");
				 printf("\n\nExample: imflipPm infilename.bmp outname.bmp w 8\n\n");
				 printf("\n\nExample: imflipPm infilename.bmp outname.bmp V 0\n\n");
				 printf("\n\nNothing executed ... Exiting ...\n\n");
				exit(EXIT_FAILURE);
    }
	if((NumThreads<0) || (NumThreads>MAXTHREADS)){
            printf("\nNumber of threads must be between 0 and %u... \n",MAXTHREADS);
            printf("\n'1' means Pthreads version with a single thread\n");
            printf("\nYou can also specify '0' which means the 'serial' (non-Pthreads) version... \n\n");
			 printf("\n\nNothing executed ... Exiting ...\n\n");
            exit(EXIT_FAILURE);
	}
	if(NumThreads == 0){
		printf("\nExecuting the serial (non-Pthreaded) version ...\n");
	}else{
		printf("\nExecuting the multi-threaded version with %li threads ...\n",NumThreads);
	}
	switch(Flip)
	{
		case '0' : MTFlipFunc = MTFlipHM; FlipFunc=FlipImageH; strcpy(FlipType,"Horizontal HM"); break;

		case '1' : MTFlipFunc = MTFlipHMC; FlipFunc=FlipImageH;strcpy(FlipType,"Horizontal HMC");break;
		
		case 'W' : MTFlipFunc = MTFlipVM; FlipFunc=FlipImageV; strcpy(FlipType,"MTFlipVM"); break;
		case '2' : MTFlipFunc = MTFlipVM2; FlipFunc=FlipImageV; strcpy(FlipType,"MTFlipVM2"); break;
		case '3' : MTFlipFunc = MTFlipVM3; FlipFunc=FlipImageV; strcpy(FlipType,"MTFlipVM3"); break;
		
		
		
		
		
	}



	TheImage = ReadBMP(argv[1]);          
	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
	
    if(NumThreads >0)
	{
		pthread_attr_init(&ThAttr);
		pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
		for(a=0; a<REPS; a++)
		{
			for(i=0; i<NumThreads; i++)
			{
				ThParam[i] = i;
				ThErr = pthread_create(&ThHandle[i], &ThAttr, MTFlipFunc, (void *)&ThParam[i]);
				if(ThErr != 0)
				{
					printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
					exit(EXIT_FAILURE);
				}
			}
			for(i=0; i<NumThreads; i++)
			{
				pthread_join(ThHandle[i], NULL);
			}
		}
	}else{
		for(a=0; a<REPS; a++)
		{
			(*FlipFunc)(TheImage);
		}
	}

	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;
	
    //merge with header and write to file
    WriteBMP(TheImage, argv[2]);
	
 	// free() the allocated memory for the image
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); }
	free(TheImage);
   
    printf("\n\nTotal execution time: %9.4f ms.  ",TimeElapsed);
	if(NumThreads>1) printf("(%9.4f ms per thread).  ",TimeElapsed/(double)NumThreads);
	printf("\n\nFlip Type =  '%s'",FlipType);
    printf("\n (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
    
    return (EXIT_SUCCESS);
}
