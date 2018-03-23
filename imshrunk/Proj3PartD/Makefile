sort		:	sort.c
				gcc sort.c -o sort -lm -Werror

sortG		:	sortG.cu
				nvcc sortG.cu -o sortG -lm -Werror

imshrunkG	:	imshrunkG.cu ImageStuff.c ImageStuff.h
				nvcc imshrunkG.cu ImageStuff.c -o imshrunkG -lpthread -lm -Werror


