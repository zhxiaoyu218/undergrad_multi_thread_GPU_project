#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <malloc.h>


#define REPS 	     1

void SortChunkI(int ChunkSize, int Chunk_ID,    signed int *ary);
void SortChunkL(int ChunkSize, int Chunk_ID,    signed  long long *ary);
void SortChunkF(int ChunkSize, int Chunk_ID,    float *ary);
void SortChunkD(int ChunkSize, int Chunk_ID,    double  *ary);

void MergeSortI(int *list, long length);
void MergeSortL(long long *list, long length);
void MergeSortF(float *list, long length);
void MergeSortD(double *list, long length);


int main(int argc, char** argv)
{

    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;

	long				a,i;
	long 				NumOfChunk;
	
	long        		HowMany;
	int 				ChunkSize;
	char       			Type;
	
	signed int 		    *InputArrayI,*SortedArrayI;
	signed long long    *InputArrayL,*SortedArrayL;
	float       		*InputArrayF,*SortedArrayF;
	double				*InputArrayD,*SortedArrayD;
	
	
	if(argc != 4)
	{
		printf("\n  Argument is not correct \n\n");
		printf("Nothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
		
	}
	
	
	HowMany 	= atoi(argv[1])  *1024 *1024;
	ChunkSize   = atoi(argv[3]);
	Type	    = toupper(argv[2][0]);	
	
	NumOfChunk  = HowMany/ChunkSize;
	

	// inital rand number
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

	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);	
	
	//// sort
	for(a=0; a<REPS; a++)
	{
		switch(Type)
		{
			case 'I':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkI(ChunkSize, i, InputArrayI);					
					}
					printf(" \n");
					for(i=0;i<HowMany;i++)
					{
						printf("%i \n", InputArrayI[i]);
						if((i+1)%4 ==0){printf(" \n"););}
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
					printf(" \n");
					for(i=0;i<HowMany;i++)
					{
						printf("%lli \n", InputArrayI[i]);
						if((i+1)%4 ==0){printf(" \n"););}
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
						printf("%f \n", InputArrayI[i]);
						if((i+1)%4 ==0){printf(" \n"););}
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
						printf("%lf \n", InputArrayI[i]);
						if((i+1)%4 ==0){printf(" \n"););}
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
				printf("%-i \n",SortedArrayI[i]);
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
		break;
//*******************************************
		case 'L':			
					free(InputArrayL);
					free(SortedArrayL);
		break;
//*******************************************		
		case 'F':
					free(InputArrayF);
					free(SortedArrayF);
		break;
//*******************************************		
		case 'D':
					free(InputArrayD);
					free(SortedArrayD);
		break;
	}

    return (EXIT_SUCCESS);
}

void SortChunkI(int ChunkSize, int Chunk_ID,   signed int *ary)
{
	long i,j;
	long sp;
	int temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = 0; i< (ChunkSize-1); i++)
	{
		for(j = sp; j< ((ChunkSize+sp)-1-i); j++)
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
void SortChunkL(int ChunkSize, int Chunk_ID,  signed long long *ary)
{
	long i,j;
	long sp;
	long long temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = 0; i< (ChunkSize-1); i++)
	{
		for(j = sp; j< ((ChunkSize+sp)-1-i); j++)
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
void SortChunkF(int ChunkSize, int Chunk_ID,    float *ary)
{
	long i,j;
	long sp;
	float temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = 0; i< (ChunkSize-1); i++)
	{
		for(j = sp; j< ((ChunkSize+sp)-1-i); j++)
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
void SortChunkD(int ChunkSize, int Chunk_ID,    double *ary)
{
	long i,j;
	long sp;
	double temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = 0; i< (ChunkSize-1); i++)
	{
		for(j = sp; j< ((ChunkSize+sp)-1-i); j++)
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