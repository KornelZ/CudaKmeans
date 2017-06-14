#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include "utility.cuh"


char* readArgs(int argc, char **args)
{
	if(argc != MAX_ARGC) {
		return NULL;
	}
	//return file name
	printf("%s\n", args[1]);
	return args[1];
}

int logResult(char *logfile, Result *res)
{
	FILE *log = fopen(logfile, "a+");
	if(log == NULL) {
		log = fopen(logfile, "w");
		if(log == NULL) {
			return 1;
		}
	}

	fprintf(log, "========================\n");
	fprintf(log, "Gpu and cc: %s %d.%d\n", res->gpuName, res->majorVersion, res->minorVersion);
	fprintf(log, "Input size: %d, number of clusters: %d\n", res->inputSize, res->clusterNum);
	fprintf(log, "Ending error: %f, threshold: %f\n", res->delta, res->threshold);
	fprintf(log, "Needed iterations: %d\n", res->neededIterations);
	fprintf(log, "Block num: %d, threads per block: %d\n", res->blockNum, res->threads);
	fprintf(log, "Gpu time: %f\n", res->gpuTime);

	if(fclose(log) == EOF) {
		return 1;
	}
	return 0;
}

int writeEncoding(int *amount, char *text, int outputSize, char *filename)
{
	char *fname = (char*) malloc(sizeof(char) * (strlen(filename) + 5));
	if(fname == NULL) {
		return 1;
	}
	*fname = '\0';

	FILE *enc = fopen(strcat(strcat(fname, filename), ".rle") , "a+");
	if(enc == NULL) {
		return 1;
	}

	for(int i = 0; i < outputSize; i++) {
		fwrite(&amount[i + 1], sizeof(int), 1, enc);
		fwrite(&text[i], sizeof(char), 1, enc);
	}

	if(fclose(enc)) {
		return 1;
	}
	return 0;
}

long getFileSize(FILE *file)
{
	if(file == NULL) {
		return 0;
	}
	if(fseek(file, 0L, SEEK_END)) {
		return 0;
	}
	long size = ftell(file);
	rewind(file);

	return size + 1;
}

void getChunkSize(dim3 grid, dim3 block, int *chunkSize, int *chunkCount, int inputSize)
{
	*chunkSize = grid.x * grid.y * grid.z * block.x * block.y * block.z;
	*chunkCount = (int)ceil((double)inputSize / *chunkSize);
}

int handleError(cudaError_t error, char *errorMsg)
{
	if(error != cudaSuccess) {
		fprintf(stderr, "%s; error: %s\n", errorMsg, cudaGetErrorString(error));
		return 1;
	}
	return 0;
}

cudaError_t startTimer(cudaEvent_t *start, cudaEvent_t *stop)
{
	cudaError_t error;
	if(handleError(error = cudaEventCreate(start), "cudaEventCreate start failed")) { return error; }
	if(handleError(error = cudaEventCreate(stop), "cudaEventCreate stop failed")) { return error; }
	if(handleError(error = cudaEventRecord(*start), "cudaEventRecord start failed")) { return error; }
	return error;
}

cudaError_t stopTimer(cudaEvent_t *start, cudaEvent_t *stop, char *eventName, float *totalTime)
{
	cudaError_t error;
	float milliseconds = 0;
	if(handleError(error = cudaEventRecord(*stop), "cudaEventRecord stop failed")) { return error; }
	if(handleError(error = cudaEventSynchronize(*stop), "cudaEventSynchronize failed")) { return error; }
	if(handleError(error = cudaEventElapsedTime(&milliseconds, *start, *stop), "cudaEventElapsedTime failed")) { return error; }
	printf("%f elapsed time : %s\n", milliseconds, eventName);

	*totalTime += milliseconds;
	return error;
}

cudaError_t checkAvailableMemory(size_t dataMemory, size_t clusterMemory)
{
	size_t freeMem = 0, totalMem = 0;
	if(handleError(cudaMemGetInfo(&freeMem, &totalMem), "cudaMemGetInfo failed")) {
		return cudaError::cudaErrorMemoryAllocation;
	}
	size_t allocatedMem = dataMemory * (sizeof(dim3) + sizeof(int)) + clusterMemory * (sizeof(dim3) * 2 + sizeof(int));

	printf("free memory: %d, memory to be allocated: %d, left memory: %d\n", freeMem, allocatedMem, freeMem - allocatedMem);
	if(freeMem < allocatedMem + MEGABYTE ) {
		return cudaError::cudaErrorMemoryAllocation;
	}

	return cudaError::cudaSuccess;
}

bool checkSharedMemory(size_t available, size_t required)
{
	if(available < sizeof(dim3) * required) {
		printf("Available shared memory per block: %u, required: %u\n", available, required * sizeof(dim3));
		return false;
	}
	return true;
}

cudaDeviceProp showGpu()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Major revision number:         %d\n",  prop.major);
    printf("Minor revision number:         %d\n",  prop.minor);
    printf("Name:                          %s\n",  prop.name);
    printf("Total global memory:           %u\n",  prop.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  prop.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  prop.regsPerBlock);
    printf("Warp size:                     %d\n",  prop.warpSize);
    printf("Maximum memory pitch:          %u\n",  prop.memPitch);
    printf("Maximum threads per block:     %d\n",  prop.maxThreadsPerBlock);
    for (int i = 0; i < 3; i++) {
		printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
	}
    for (int i = 0; i < 3; i++) {
		printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);
	}
    printf("Clock rate:                    %d\n",  prop.clockRate);
    printf("Total constant memory:         %u\n",  prop.totalConstMem);
    printf("Texture alignment:             %u\n",  prop.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  prop.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

	return prop;
}
