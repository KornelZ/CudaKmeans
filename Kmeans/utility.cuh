#include "cuda_runtime.h"
#include <cstdio>
#include "device_launch_parameters.h"

#define MEGABYTE 1048576
#define MAX_ARGC 2

typedef struct result {
	float gpuTime;
	float delta;
	float threshold;
	int inputSize;
	int neededIterations;
	int blockNum;
	int threads;
	int clusterNum;
	int majorVersion;
	int minorVersion;
	char *gpuName;
} Result;

char* readArgs(int argc, char **args);
void readFile(FILE *file, char **buffer, int size);
int logResult(char *logfile, Result *res);
long getFileSize(FILE *file);
void getChunkSize(dim3 grid, dim3 block, int *chunkSize, int *chunkCount, int inputSize); 
int writeEncoding(int *amount, char *text, int outputSize, char *filename);

int handleError(cudaError_t error, char *errorMsg);

cudaError_t startTimer(cudaEvent_t *start, cudaEvent_t *stop);
cudaError_t stopTimer(cudaEvent_t *start, cudaEvent_t *stop, char *eventName, float *totalTime);

cudaError_t checkAvailableMemory(size_t dataMemory, size_t clusterMemory);
bool checkSharedMemory(size_t available, size_t required);
cudaDeviceProp showGpu();
