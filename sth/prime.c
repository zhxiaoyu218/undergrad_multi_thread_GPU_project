#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>

# define REPS 	     1
# define NumOfSet 8192

	long numbers[NumOfSet];
	long Primes[NumOfSet];
	int PrimeIndexes[NumOfSet];

int main(int argc, char** argv)
{
	int i,j,y;
	int PrimeCount;
	int temp;


	int flag;

	struct timeval 		t;
	double         		StartTime, EndTime;
	double         		TimeElapsed;

	FILE *fp,*ff;

	fp = fopen(argv[1], "r");


	if (fp == NULL)
	{
		printf("error,file cannot be opened");
		exit(0);
	}
	printf("\nInputfile:  %s \n\nOutputfile: %s\n\n",argv[1],argv[2]);
	for (i = 0; i < NumOfSet; i++)          //load number from file
	{
		fscanf(fp, "%ld\n", &numbers[i]);
	}
	fclose(fp);

	gettimeofday(&t, NULL);
	StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);

	
	for (i = 0; i < NumOfSet; i++)    
	{
		flag = 1;
		temp = (int)sqrt(numbers[i]);
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

	
	gettimeofday(&t, NULL);
	EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;

	
	ff = fopen(argv[2], "w");
	
	for (y = 0; y < PrimeCount ; y++)
	{
		fprintf(ff, "%ld: %ld\n", PrimeIndexes[y],Primes[y]);
	}
	
	printf("\n\nTotal execution time: %9.4f ms ",TimeElapsed);


	fclose(ff);


	getchar();
	return 0;
}