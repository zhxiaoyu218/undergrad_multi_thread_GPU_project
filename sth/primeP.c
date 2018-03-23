#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <ctype.h>



# define NumOfSet 8192
# define MAXTHREADS   128

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...



unsigned long numbers[NumOfSet];
unsigned long Primes[NumOfSet];
int PrimeIndexes[NumOfSet];       //to be decide how to archieve
int PrimeCount = 0;

pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes

void *MTPrime(void* tid) 
{
	int i,j,flag,temp;

	long ts = *((int *)tid);       	// My thread ID is stored here
	ts *= ceil(NumOfSet / NumThreads);			// start index
	long te = ts + NumOfSet / NumThreads - 1; 	// end index

	for (i = ts; i < te; i++)    
	{
		flag = 1;
		temp = (int)sqrt(numbers[i])+1;
		for (j = 2; j <= temp; j++)
		{
			if (numbers[i] % j == 0)
			{
				flag = 0;
				break;
			}
		}

		if (flag == 1)
		{
			Primes[PrimeCount] = numbers[i];
			PrimeIndexes[PrimeCount] = i+1;
			PrimeCount++;

		}

		
	}	

	pthread_exit(0);

}


int main(int argc, char** argv)
{

	int i,j,y,ThErr;
	
	int flag;
	NumThreads = atoi(argv[3]);
	
	struct timeval 		t;
	double         		StartTime, EndTime;
	double         		TimeElapsed;
	
	FILE *fp,*ff;
	fp = fopen(argv[1], "r");

	
	if (fp == NULL)
	{
		printf("\n error,file cannot be opened\n");
		exit(0);
	}
	printf("\nThread:     %i\n\nInputfile:  %s \n\nOutputfile: %s\n\n",NumThreads,argv[1],argv[2]);


	for (i = 0; i < NumOfSet; i++)          //load number from file
	{
		fscanf(fp, "%lu\n", &numbers[i]);
	}
	
	fclose(fp);
	
	gettimeofday(&t, NULL);
	StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);

	pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
	
		for (j = 0; j<NumThreads; j++) 
		{			
			ThParam[j] = j;
			ThErr = pthread_create(&ThHandle[j], &ThAttr, MTPrime, (void *)&ThParam[j]);
			if (ThErr != 0) 
			{
				printf("\nThread Creation Error %d. Exiting abruptly... \n", ThErr);
				exit(EXIT_FAILURE);
			}
			
		}
		
		for (j = 0; j<NumThreads; j++) {
			pthread_join(ThHandle[j], NULL);
		}
	
	
	gettimeofday(&t, NULL);
	EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	
	ff = fopen(argv[2], "w");
	for (y = 0; y < PrimeCount ; y++)
	{
		fprintf(ff, "%lu: %lu \n", PrimeIndexes[y],Primes[y]);
	}
	
	printf("\n\nTotal execution time: %9.4f ms ",TimeElapsed);
	
	fclose(ff);

	getchar();
	return 0;





}