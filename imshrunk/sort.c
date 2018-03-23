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

void SortChunkI(int ChunkSize, int Chunk_ID,    int ary[]);
void SortChunkL(int ChunkSize, int Chunk_ID,    long long ary[]);
void SortChunkF(int ChunkSize, int Chunk_ID,    float ary[]);
void SortChunkD(int ChunkSize, int Chunk_ID,    double  ary[]);

void MergeI(int InputArr[],int Arr[],int start,int mid,int end);
void MergeSortI(int InputArr[],int Arr[],int start,int end);

void MergeL(long long InputArr[],long long Arr[],int start,int mid,int end);
void MergeSortL(long long InputArr[],long long Arr[],int start,int end);

void MergeF(float InputArr[],float Arr[],int start,int mid,int end);
void MergeSortF(float InputArr[],float Arr[],int start,int end);

void MergeD(double InputArr[],double Arr[],int start,int midIndex,int end);
void MergeSortD(double InputArr[],double Arr[],int start,int end);



int main(int argc, char** argv)
{

    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;

	int 				a,i;
	long 		NumOfChunk;
	
	long         HowMany;
	int 		ChunkSize;
	char        Type;
	
	int 		*InputArrayI,*SortedArrayI;
	long long   *InputArrayL,*SortedArrayL;
	float       *InputArrayF,*SortedArrayF;
	double		*InputArrayD,*SortedArrayD;
	
	
	if(argc != 4)
	{
		printf("\n  Argument is not correct \n\n");
		printf("Nothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
		
	}
	
	
	HowMany 	= atoi(argv[1]) ;
	ChunkSize   = atoi(argv[3]);
	Type	    = toupper(argv[2][0]);	
	
	NumOfChunk  = HowMany/ChunkSize;
	
	
	InputArrayI  = (int *)malloc(HowMany * sizeof(int));
	InputArrayL  = (long long *)malloc(HowMany * sizeof(long long )); 
	InputArrayF  = (float *)malloc(HowMany * sizeof(float));
	InputArrayD  = (double *)malloc(HowMany * sizeof(double));

	
	
	for(i=0;i<HowMany;i++)
	{InputArrayI[i] = rand()*rand();}
	
	for(i=0;i<HowMany;i++)
	{
	InputArrayL[i]   =(long long )InputArrayI[i] <<32  |  ((long long )(rand()+rand()));

	}
	// inital rand number
	switch(Type)
	{
		case 'I':
				//InputArrayI  = (int *)malloc(HowMany * sizeof(int));
				SortedArrayI = (int *)malloc(HowMany * sizeof(int));
		break;
//*******************************************
		case 'L':
			//	InputArrayL   = (long long *)malloc(HowMany * sizeof(long long )); 
				SortedArrayL  = (long long *)malloc(HowMany * sizeof(long long ));	
		break;
//*******************************************		
		case 'F':
			//	InputArrayF  = (float *)malloc(HowMany * sizeof(float));
				SortedArrayF = (float *)malloc(HowMany * sizeof(float));
				for(i=0;i<HowMany;i++)
				{
					InputArrayF[i]  =   *(float*)&InputArrayI[i];
				}
			//	while(isnan(InputArrayF))
			//	{
					//	InputArrayI = rand()+rand();
			//			InputArrayF  =   *(float*)&(InputArrayI);
			//	}
		break;
//*******************************************		
		case 'D':
			//	InputArrayD  = (double *)malloc(HowMany * sizeof(double));
				SortedArrayD = (double *)malloc(HowMany * sizeof(double));
				for(i=0;i<HowMany;i++)
				{
					InputArrayD[i]  =   *(double*)&InputArrayL[i];
				}
				//while(isnan(InputArrayD))
				//{
				//		InputArrayD  =   *(double*)&(InputArrayL);
				//}
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
					MergeSortI(InputArrayI,SortedArrayI,0,HowMany-1);
			break;
//*******************************************
			case 'L':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkL(ChunkSize, i, InputArrayL);					
					}
					MergeSortL(InputArrayL,SortedArrayL,0,HowMany-1);
			break;
//*******************************************		
			case 'F':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkF(ChunkSize, i, InputArrayF);					
					}
					MergeSortF(InputArrayF,SortedArrayF,0,HowMany-1);
			break;
//*******************************************		
			case 'D':
					for(i=0;i<NumOfChunk;i++)
					{
						SortChunkD(ChunkSize, i, InputArrayD);					
					}
					MergeSortD(InputArrayD,SortedArrayD,0,HowMany-1);
			break;

		}	
	}
    
	gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;

    printf("\n\nExecution time:%10.4f ms  ",TimeElapsed);

	
	
	switch(Type)
		{
			case 'I':
				printf("\n");
				for(i=0; i<HowMany; i++)
				{
					printf("  %i   ;   %i \n",InputArrayI[i],SortedArrayI[i]);
				}
				
			break;
//*******************************************
			case 'L':
				printf("\n");
				for(i=0; i<HowMany; i++)
				{
					printf("  %lli   ;   %lli \n",InputArrayL[i],SortedArrayL[i]);
				}
			break;
//*******************************************		
			case 'F':
				printf("\n");
				for(i=0; i<HowMany; i++)
				{
					printf("  %.4f   ;   %.4f\n",InputArrayF[i],SortedArrayF[i]);
				}
			break;
//*******************************************		
			case 'D':
				printf("\n");
				for(i=0; i<HowMany; i++)
				{
					printf("  %.4lf   ;   %.4lf\n",InputArrayD[i],SortedArrayD[i]);
				}
			break;

		}
	
	printf("\n **************************\n");
		switch(Type)
	{
		case 'I':
					free(InputArrayI);
					free(InputArrayL);
					free(InputArrayF);
					free(InputArrayD);
						free(SortedArrayI);
		break;
//*******************************************
		case 'L':
					free(InputArrayI);
					free(InputArrayL);
					free(InputArrayF);
					free(InputArrayD);
						free(SortedArrayL);
		break;
//*******************************************		
		case 'F':
					free(InputArrayI);
					free(InputArrayL);
					free(InputArrayF);
					free(InputArrayD);
						free(SortedArrayF);
		break;
//*******************************************		
		case 'D':
					free(InputArrayI);
					free(InputArrayL);
					free(InputArrayF);
					free(InputArrayD);
						free(SortedArrayD);
		break;
	}
	printf("\n *********222222222222222222**********\n");
    return (EXIT_SUCCESS);
}

void SortChunkI(int ChunkSize, int Chunk_ID,    int ary[])
{
	int i,j;
	int sp;
	int temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< ChunkSize; i++)
	{
		for(j = sp; j< (ChunkSize-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}	
	 
	for (i=sp;i<ChunkSize;i++)  
    {  
            printf("    %i \n",ary[i]);  
    }  
	return;
}
void SortChunkL(int ChunkSize, int Chunk_ID,    long long ary[])
{
	int i,j;
	int sp;
	long long temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< ChunkSize; i++)
	{
		for(j = sp; j< (ChunkSize-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}	
	
	
	for (i=sp;i<ChunkSize;i++)  
    {  
            printf("    %lli \n",ary[i]);  
    }  
	return;		
}
void SortChunkF(int ChunkSize, int Chunk_ID,    float ary[])
{
	int i,j;
	int sp;
	float temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< ChunkSize; i++)
	{
		for(j = sp; j< (ChunkSize-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}	

	for (i=sp;i<ChunkSize;i++)  
    {  
            printf("    %f \n",ary[i]);  
    }  
	return;	
}
void SortChunkD(int ChunkSize, int Chunk_ID,    double  ary[] )
{
	int i,j;
	int sp;
	double temp;
	
	sp = Chunk_ID * ChunkSize;

	for(i = sp; i< ChunkSize; i++)
	{
		for(j = sp; j< (ChunkSize-1); j++)
		{
			if(ary[j] > ary[j+1])
			{
				temp       = ary[j+1];
				ary[j+1]   = ary[j];
				ary[j]     = temp;
			}
		}
	}


	for (i=sp;i<ChunkSize;i++)  
    {  
            printf("    %lf \n",ary[i]);  
    }  	
	return;		
}



void MergeI(int InputArr[],int Arr[],int start,int mid,int end)
{
    int i = start,j=mid+1,k = start;
    while(i!=mid+1 && j!=end+1)
    {
        if(InputArr[i]>InputArr[j])
            Arr[k++] = InputArr[i++];
        else
            Arr[k++] = InputArr[j++];
    }
    while(i!=mid+1)
        Arr[k++] = InputArr[i++];
    while(j!=end+1)
        Arr[k++] = InputArr[j++];
  //  for(i=start;i<=end;i++)  InputArr[i] = Arr[i];
  return;
}
void MergeSortI(int InputArr[],int Arr[],int start,int end)
{
    int mid;
    if( start < end )
    {
        mid=( start + end ) / 2;
        MergeSortI(InputArr, Arr, start, mid);
        MergeSortI(InputArr, Arr, mid+1, end);
        MergeI(InputArr, Arr, start, mid,end);
    }
	return;
}

void MergeL(long long InputArr[],long long Arr[],int start,int mid,int end)
{
    int i = start,j=mid+1,k = start;
    while(i!=mid+1 && j!=end+1)
    {
        if(InputArr[i]>InputArr[j])
            Arr[k++] = InputArr[i++];
        else
            Arr[k++] = InputArr[j++];
    }
    while(i!=mid+1)
        Arr[k++] = InputArr[i++];
    while(j!=end+1)
        Arr[k++] = InputArr[j++];
  //  for(i=start;i<=end;i++)  InputArr[i] = Arr[i];
  return;
}
void MergeSortL(long long InputArr[],long long Arr[],int start,int end)
{
    int mid;
    if( start < end )
    {
        mid=( start + end ) / 2;
        MergeSortL(InputArr, Arr, start, mid);
        MergeSortL(InputArr, Arr, mid+1, end);
        MergeL(InputArr, Arr, start, mid,end);
    }
	return;
}

void MergeF(float InputArr[],float Arr[],int start,int mid,int end)
{
    int i = start,j=mid+1,k = start;
    while(i!=mid+1 && j!=end+1)
    {
        if(InputArr[i]>InputArr[j])
            Arr[k++] = InputArr[i++];
        else
            Arr[k++] = InputArr[j++];
    }
    while(i!=mid+1)
        Arr[k++] = InputArr[i++];
    while(j!=end+1)
        Arr[k++] = InputArr[j++];
  //  for(i=start;i<=end;i++)  InputArr[i] = Arr[i];
  return;
}
void MergeSortF(float InputArr[],float Arr[],int start,int end)
{
    int mid;
    if( start < end )
    {
        mid=( start + end ) / 2;
        MergeSortF(InputArr, Arr, start, mid);
        MergeSortF(InputArr, Arr, mid+1, end);
        MergeF(InputArr, Arr, start, mid,end);
    }
	return;
}

void MergeD(double InputArr[],double Arr[],int start,int mid,int end)
{
    int i = start,j=mid+1,k = start;
    while(i!=mid+1 && j!=end+1)
    {
        if(InputArr[i]>InputArr[j])
            Arr[k++] = InputArr[i++];
        else
            Arr[k++] = InputArr[j++];
    }
    while(i!=mid+1)
        Arr[k++] = InputArr[i++];
    while(j!=end+1)
        Arr[k++] = InputArr[j++];
  //  for(i=start;i<=end;i++)  InputArr[i] = Arr[i];
  return;
}
void MergeSortD(double InputArr[],double Arr[],int start,int end)
{
    int mid;
    if( start < end )
    {
        mid=( start + end ) / 2;
        MergeSortD(InputArr, Arr, start, mid);
        MergeSortD(InputArr, Arr, mid+1, end);
        MergeD(InputArr, Arr, start, mid,end);
    }
	return;
}

