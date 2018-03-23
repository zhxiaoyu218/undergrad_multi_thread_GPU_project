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



#define REPS 	     1

void SortChunkI(long ChunkSize, int Chunk_ID,    signed int *ary);
void SortChunkL(long ChunkSize, int Chunk_ID,    signed  long long *ary);
void SortChunkF(long ChunkSize, int Chunk_ID,    float *ary);
void SortChunkD(long ChunkSize, int Chunk_ID,    double  *ary);

void MergeSortI(int *list, long length);
void MergeSortL(long long *list, long length);
void MergeSortF(float *list, long length);
void MergeSortD(double *list, long length);


__global__ void SortChunkGI(signed int ary[], long ChunkSize)
{
	unsigned long i,j;
	unsigned long sp;
	int temp;
	
	sp =  (blockIdx.x * blockDim.x +threadIdx.x)*ChunkSize; 	
	printf("\n start:%li    end:%li   Chunksize: %li   S_ary: %i E_ary:%i\n",sp,ChunkSize+sp-1,ChunkSize,ary[sp],ary[ChunkSize+sp-1]);
	for(i = 0; i< ChunkSize; i++)
	{
		for(j = sp; j< (ChunkSize+sp-1-i); j++)
		{
			if( (ary[j]) > (ary[j+1]))
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}
}


int main(int argc, char** argv)
{
//****************    HOST   variable ******************
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;

	long				a,i;
	long 				NumOfChunk;
	
	long        		HowMany;
	long 				ChunkSize;
	char       			Type;
	
	signed int 		    *InputArrayI,*SortedArrayI;
	signed long long    *InputArrayL,*SortedArrayL;
	float       		*InputArrayF,*SortedArrayF;
	double				*InputArrayD,*SortedArrayD;
	
	long 				BlockSize;
	long				NumOfBlock;
//****************    GPU   variable ******************	
	signed int 		    *InputArrayG_I,*SortedArrayG_I;
	signed long long    *InputArrayG_L,*SortedArrayG_L;
	float       		*InputArrayG_F,*SortedArrayG_F;
	double				*InputArrayG_D,*SortedArrayG_D;
	
	FILE *ff = fopen("output.txt", "w");
	
	
	if(argc != 5)
	{
		printf("\n  Argument is not correct \n\n");
		printf("Nothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);	
	}
	
	
	HowMany 	= atoi(argv[1]);
	ChunkSize   = atoi(argv[3]);
	Type	    = toupper(argv[2][0]);	
	BlockSize   = atoi(argv[4]);
	
	NumOfChunk  = HowMany/(ChunkSize);
	NumOfBlock  = HowMany/((ChunkSize * BlockSize));
	
	printf("\nElement type   :        %c\n",Type);
	printf("BlockSize      :        %i\n",BlockSize);
	printf("Total Block    :        %i\n",NumOfBlock);
	printf("Total Element  :        %i\n\n\n\n",NumOfBlock*ChunkSize*BlockSize);
	
	
	
// HOST*************inital rand number
	switch(Type)
	{
		case 'I':
				InputArrayI  = (signed int *)malloc(HowMany * sizeof(signed int));	
				SortedArrayI = (signed  int *)malloc(HowMany * sizeof(signed int));	
				for(i=0;i<HowMany;i++)
				{
					InputArrayI[i] = (  ((-1)^i)*rand()   );
				}				
		break;
//*******************************************
		case 'L':	
				InputArrayL  = (signed long long *)malloc(HowMany * sizeof(signed long long )); 
				SortedArrayL  = (signed long long *)malloc(HowMany * sizeof(signed long long ));
				for(i=0;i<HowMany;i++)
				{
					InputArrayL[i]  =(long long )((  ((-1)^i)*rand())<<32   |  rand()  );
				}
		break;
//*******************************************		
		case 'F':
				InputArrayF  = (float *)malloc(HowMany * sizeof(float));
				SortedArrayF = (float *)malloc(HowMany * sizeof(float));
				
				int my_random_int;
				for(i=0;i<HowMany;i++)
				{
					my_random_int = ((-1)^i)*rand() ;
					InputArrayF[i]  =   *(float*)&my_random_int;
					
					if(isnan(InputArrayF[i])){i--;}
				}
		break;
//*******************************************		
		case 'D':
				InputArrayD  = (double *)malloc(HowMany * sizeof(double));
				SortedArrayD = (double *)malloc(HowMany * sizeof(double));
				long long int my_random_long;
				for(i=0;i<HowMany;i++)
				{
					my_random_long = (long long )(( ((-1)^i)*rand()<<32)   | rand()  );
					InputArrayD[i]  =   *(double*)&my_random_long;				
					
					if(isnan(InputArrayD[i])){i--;}
				}
		break;

	}
// GPU*********** inital GPU and transfer data HtoD
	switch(Type)
	{
		case 'I':
				cudaMalloc ((signed int **)&InputArrayG_I, HowMany*sizeof(signed int));
				cudaMalloc ((signed int **)&SortedArrayG_I, HowMany*sizeof(signed int));
				cudaMemcpy (InputArrayG_I, InputArrayI, HowMany*sizeof(signed int), cudaMemcpyHostToDevice);
		break;
//*******************************************
		case 'L':	
				cudaMalloc ((signed long long **)&InputArrayG_L, HowMany* sizeof(signed long long ));
				cudaMalloc ((signed long long **)&SortedArrayG_L, HowMany* sizeof(signed long long ));
				cudaMemcpy (InputArrayG_L, InputArrayL, HowMany, cudaMemcpyHostToDevice);
		break;
//*******************************************		
		case 'F':
				cudaMalloc ((float **)&InputArrayG_F, HowMany);
				cudaMalloc ((float **)&SortedArrayG_F, HowMany);
				cudaMemcpy (InputArrayG_F, InputArrayF, HowMany, cudaMemcpyHostToDevice);
		break;
//*******************************************		
		case 'D':
				cudaMalloc ((double **)&InputArrayG_D, HowMany);
				cudaMalloc ((double **)&SortedArrayG_D, HowMany);
				cudaMemcpy (InputArrayG_D, InputArrayD, HowMany, cudaMemcpyHostToDevice);
		break;
	}
	
	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);	
	
//*******************         sort   ***************
	for(a=0; a<REPS; a++)
	{
		switch(Type)
		{
			case 'I':
					
				//				tot block     element per block
					SortChunkGI<<< NumOfBlock, BlockSize>>> (InputArrayG_I,ChunkSize);					
					cudaMemcpy (InputArrayI, InputArrayG_I, HowMany*sizeof(signed int), cudaMemcpyDeviceToHost);	
					
					for(i=0;i<HowMany;i++)
					{
						fprintf(ff, "%i \n", InputArrayI[i]);
						if((i+1)%4 ==0){fprintf(ff, " \n");}
					}
					for(i=0;i<HowMany;i++)
					{
						SortedArrayI[i]=InputArrayI[i];
					}										
					 MergeSortI(SortedArrayI, HowMany);
			break;
//*******************************************
			case 'L':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkL(ChunkSize, i, InputArrayL);					
					}
					for(i=0;i<HowMany;i++)
					{
						SortedArrayL[i]=InputArrayL[i];
					}
					 MergeSortL(SortedArrayL, HowMany);
			break;
//*******************************************		
			case 'F':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkF(ChunkSize, i, InputArrayF);					
					}
					for(i=0;i<HowMany;i++)
					{
						SortedArrayF[i]=InputArrayF[i];
					}
					MergeSortF(SortedArrayF, HowMany);
			break;
//*******************************************		
			case 'D':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkD(ChunkSize, i, InputArrayD);					
					}
					for(i=0;i<HowMany;i++)
					{
						SortedArrayD[i]=InputArrayD[i];
					}
					MergeSortD(SortedArrayD, HowMany);
			break;

		}	
	}
    
	gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;

    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);

// print result	
	switch(Type)
	{
		case 'I':
			printf("\n");
			for(i=0; i<HowMany; i++)
			{
				//printf("%-i \n",SortedArrayI[i]);
			}
		break;
//*******************************************
		case 'L':
			printf("\n");
			for(i=0; i<HowMany; i++)
			{
				printf("%-lli \n",SortedArrayL[i]);
			}
		break;
//*******************************************		
		case 'F':
			printf("\n");
			for(i=0; i<HowMany; i++)
			{
				printf("%-.4f\n",SortedArrayF[i]);
			}
		break;
//******************************************		
		case 'D':
			printf("\n");
			for(i=0; i<HowMany; i++)
			{
				printf("%-.4lf\n",SortedArrayD[i]);
			}
		break;
	}
	
//free memoary
	switch(Type)
	{
		case 'I':
					free(InputArrayI);
					free(SortedArrayI);
					cudaFree(InputArrayG_I);
					cudaFree(SortedArrayG_I);
		break;
//*******************************************
		case 'L':			
					free(InputArrayL);
					free(SortedArrayL);
					cudaFree(InputArrayG_L);
					cudaFree(SortedArrayG_L);
		break;
//*******************************************		
		case 'F':
					free(InputArrayF);
					free(SortedArrayF);
					cudaFree(InputArrayG_F);
					cudaFree(SortedArrayG_F);
		break;
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

void SortChunkI(long ChunkSize, int Chunk_ID,   signed int *ary)
{
	long i,j;
	long sp;
	int temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< (ChunkSize+sp); i++)
	{
		for(j = sp; j< ((ChunkSize+sp)-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}
	
	for(i = sp; i< (ChunkSize+sp); i++)
	{
		
		printf("\n %i",ary[i]);
	}
			printf("\n \n");
	return;
}
void SortChunkL(long ChunkSize, int Chunk_ID,  signed long long *ary)
{
	long i,j;
	long sp;
	long long temp;
	
	sp = Chunk_ID * ChunkSize;

		for(i = sp; i< (ChunkSize+sp); i++)
	{
		for(j = sp; j< (ChunkSize+sp-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}	
	
		for(i = sp; i< (ChunkSize+sp); i++)
	{
		
		printf("\n %lli",ary[i]);
	}
			printf("\n \n");


	return;		
}
void SortChunkF(long ChunkSize, int Chunk_ID,    float *ary)
{
	long i,j;
	long sp;
	float temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< (ChunkSize+sp); i++)
	{
		for(j = sp; j< (ChunkSize+sp-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}	
		for(i = sp; i< (ChunkSize+sp); i++)
	{
		
		printf("\n %f",ary[i]);
	}
			printf("\n \n");

	return;	
}
void SortChunkD(long ChunkSize, int Chunk_ID,    double *ary)
{
	long i,j;
	long sp;
	double temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< (ChunkSize+sp); i++)
	{
		for(j = sp; j< (ChunkSize+sp); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}

		for(i = sp; i< (ChunkSize+sp); i++)
	{
		
		printf("\n %f",ary[i]);
	}
			printf("\n \n");

	
	return;		
}

 void MergeSortI(signed int *list, long length)
 {
     long i;
	 long left_min, left_max, right_min, right_max, next;
     
	signed  int *tmp = (int*)malloc(sizeof(int) * length);
 
     if (tmp == NULL)
     {
         fputs("Error: out of memory\n", stderr);
         abort();
     }
 
     for (i = 1; i < length; i *= 2) 
     {
         for (left_min = 0; left_min < length - i; left_min = right_max)
         {
             right_min = left_max = left_min + i;
             right_max = left_max + i;
 
             if (right_max > length)
                 right_max = length;
 
             next = 0;
             while (left_min < left_max && right_min < right_max)
                 tmp[next++] = list[left_min] > list[right_min] ? list[right_min++] : list[left_min++];
 
             while (left_min < left_max)
                 list[--right_min] = list[--left_max];
 
             while (next > 0)
                 list[--right_min] = tmp[--next];
			 
			 
         }
     }
     free(tmp);
	return;
 }

 void MergeSortL(signed long long *list, long length)
 {
     long i;
	 long left_min, left_max, right_min, right_max, next;
     
	signed  long long  *tmp = (long long *)malloc(sizeof(long long ) * length);
 
     if (tmp == NULL)
     {
         fputs("Error: out of memory\n", stderr);
         abort();
     }
 
     for (i = 1; i < length; i *= 2) 
     {
         for (left_min = 0; left_min < length - i; left_min = right_max)
         {
             right_min = left_max = left_min + i;
             right_max = left_max + i;
 
             if (right_max > length)
                 right_max = length;
 
             next = 0;
             while (left_min < left_max && right_min < right_max)
                 tmp[next++] = list[left_min] > list[right_min] ? list[right_min++] : list[left_min++];
 
             while (left_min < left_max)
                 list[--right_min] = list[--left_max];
 
             while (next > 0)
                 list[--right_min] = tmp[--next];
		 
         }
     }
     free(tmp);
	return;
 }
 
 void MergeSortF(float *list, long length)
 {
     long i;
	 long left_min, left_max, right_min, right_max, next;
     
	 float  *tmp = (float *)malloc(sizeof(float ) * length);
 
     if (tmp == NULL)
     {
         fputs("Error: out of memory\n", stderr);
         abort();
     }
 
     for (i = 1; i < length; i *= 2) 
     {
         for (left_min = 0; left_min < length - i; left_min = right_max)
         {
             right_min = left_max = left_min + i;
             right_max = left_max + i;
 
             if (right_max > length)
                 right_max = length;
 
             next = 0;
             while (left_min < left_max && right_min < right_max)
                 tmp[next++] = list[left_min] > list[right_min] ? list[right_min++] : list[left_min++];
 
             while (left_min < left_max)
                 list[--right_min] = list[--left_max];
 
             while (next > 0)
                 list[--right_min] = tmp[--next];
		 
         }
     }
     free(tmp);
	return;
 }

 void MergeSortD(double *list, long length)
 {
     long i;
	 long left_min, left_max, right_min, right_max, next;
     
	 double  *tmp = (double *)malloc(sizeof(double ) * length);
 
     if (tmp == NULL)
     {
         fputs("Error: out of memory\n", stderr);
         abort();
     }
 
     for (i = 1; i < length; i *= 2) 
     {
         for (left_min = 0; left_min < length - i; left_min = right_max)
         {
             right_min = left_max = left_min + i;
             right_max = left_max + i;
 
             if (right_max > length)
                 right_max = length;
 
             next = 0;
             while (left_min < left_max && right_min < right_max)
                 tmp[next++] = list[left_min] > list[right_min] ? list[right_min++] : list[left_min++];
 
             while (left_min < left_max)
                 list[--right_min] = list[--left_max];
 
             while (next > 0)
                 list[--right_min] = tmp[--next];
		 
         }
     }
     free(tmp);
	return;
 }