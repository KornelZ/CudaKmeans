#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "utility.cuh"
#include <thrust\reduce.h>
#include <thrust\execution_policy.h>
#include <thrust\device_ptr.h>
#define CLUSTER_NUM 128  //number of clusters
#define DATA_SIZE 30000000 //dataset size
#define BLOCK_SIZE 1024 //threads per block
#define LOG_FILE "log.txt" 
#define THRESHOLD 0.01 //min possible % of label changes to end the algorithm
#define MAX_ITER 30

cudaError_t kmeansInit(dim3 *data, dim3 *cluster, int *label, double threshold, long dataSize, int clusterSize, 
					   int iterations, Result * result, int blockNum);
cudaError_t kmeans(dim3 *d_data, dim3 *d_cluster, int *d_label, int *d_counter, dim3 *d_sum,
				   double threshold, long dataSize, int clusterSize, int iterations, Result *result, int blockNum);
//calculates distance between cluster and data vector
__device__ long findDistance(dim3 data, dim3 cluster)
{
	return (data.x - cluster.x) * (data.x - cluster.x) 
		 + (data.y - cluster.y) * (data.y - cluster.y) 
		 + (data.z - cluster.z) * (data.z - cluster.z);
}
//relabels data vectors, counts sum and amount of data assigned to each cluster
__global__ void clusterData(dim3 *data, dim3 *sum, int *counter, dim3 *cluster, int clusterCount, long dataSize, int *label, int *labelChanges)
{
	__shared__ dim3 sh[CLUSTER_NUM];

	for(int i = 0; i < CLUSTER_NUM; i++) {
		sh[i] = cluster[i];
	}
	int index, i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < dataSize) {
		long min = LONG_MAX, dist;
		for(int j = 0; j < clusterCount; j++) {
			if(min > (dist = findDistance(data[i], sh[j]))) {
				min = dist;
				index = j;
			}
		}
		if(label[i] != index + 1) {
			label[i] = index + 1;
			labelChanges[i] = 1;
		}

		atomicAdd(&sum[index].x, data[i].x);
		atomicAdd(&sum[index].y, data[i].y);
		atomicAdd(&sum[index].z, data[i].z);

		atomicAdd(&counter[index], 1);

	}
}
//calculates new cluster centers
__global__ void refreshClusters(dim3 *sum, dim3 *cluster, int *counter)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(counter[i] != 0) {
		cluster[i].x = sum[i].x / counter[i];
		cluster[i].y = sum[i].y / counter[i];
		cluster[i].z = sum[i].z / counter[i];
	} else {
		cluster[i].z = cluster[i].x = cluster[i].z = 0;
	}
	sum[i] = dim3(0, 0, 0);
	counter[i] = 0;
}

__global__ void clearLabelChanges(int *labelChanges)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	labelChanges[i] = 0;
}

int main(int argc, char **argv)
{
	cudaDeviceProp prop = showGpu();
	if(checkAvailableMemory(DATA_SIZE, CLUSTER_NUM) != cudaSuccess) {
		fprintf(stderr, "Not enough memory\n");
		return 1;
	}
	if(checkSharedMemory(prop.sharedMemPerBlock, CLUSTER_NUM) == false) {
		fprintf(stderr, "Not enough shared memory");
		return 1;
	}
	int dataSize = DATA_SIZE;
	int blockNum = dataSize / BLOCK_SIZE + 1;
	Result result;
	result.blockNum = blockNum;
	result.clusterNum = CLUSTER_NUM;
	result.threads = BLOCK_SIZE;
	result.threshold = THRESHOLD;
	result.inputSize = DATA_SIZE;
	result.majorVersion = prop.major;
	result.minorVersion = prop.minor;
	result.gpuName = prop.name;

	dim3 *data = (dim3*)malloc(sizeof(dim3) * dataSize);
	if(data == NULL) {
		fprintf(stderr, "Malloc failed at data\n");
		return 1;
	}
	srand(NULL);
	//assign random value of data between 0 and 9
	for(int i = 0; i < dataSize; i++) {
		data[i].x = rand() % 10;
		data[i].y = rand() % 10;
		data[i].z = rand() % 10;
	}
	dim3 cluster[CLUSTER_NUM];
	//assign random value of clusters between 0 and 9
	for(int i = 0; i < CLUSTER_NUM; i++) {
		cluster[i].x = rand() % 10;
		cluster[i].y = rand() % 10;
		cluster[i].z = rand() % 10;
	}
	int *label= (int*)malloc(sizeof(int) * dataSize);
	if(label == NULL) {
		fprintf(stderr, "Malloc failed at label\n");
		free(data);
		return 1;
	}
	double threshold = THRESHOLD;
	int iterations = MAX_ITER;

	cudaError_t cudaStatus = kmeansInit(data, cluster, label, threshold, dataSize, CLUSTER_NUM, iterations, &result, blockNum);
    if (cudaStatus != cudaSuccess) {
		free(data);
		free(label);
        fprintf(stderr, "Kmeans failed!");
        return 1;
    }
	//Print final cluster centers
	for(int i = 0; i < CLUSTER_NUM; i++) {
		printf("(%d %d %d), ", cluster[i].x, cluster[i].y, cluster[i].z);
	}
	putchar('\n');

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		free(data);
		free(label);
        return 1;
    }
	logResult(LOG_FILE, &result);
	free(data);
	free(label);
	getchar();
    return 0;
}

cudaError_t kmeans(dim3 *d_data, dim3 *d_cluster, int *d_label, int *d_counter, dim3 *d_sum,
				   double threshold, long dataSize, int clusterSize, int iterations, Result *result, int blockNum)
{
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float totalTime = 0;
	int i = 0;

	int totalChanges = 0;
	int *d_labelChanges = 0;
	bool stopped = false;
	if((cudaStatus = cudaMalloc((void**)&d_labelChanges, dataSize * sizeof(int))) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at d_labelChanges\n");
		return cudaStatus;
	}
	thrust::device_ptr<int> thr_labelChanges = thrust::device_pointer_cast<int>(d_labelChanges);
	while(i < iterations) {
		if((cudaStatus = startTimer(&start, &stop)) != cudaSuccess) { return cudaStatus; }
		//sets all label changes to 0
		clearLabelChanges<<<blockNum, BLOCK_SIZE>>>(d_labelChanges);
		if((cudaStatus = cudaGetLastError()) != cudaSuccess) {
			fprintf(stderr, "clusterData failed\n");
			free(d_labelChanges);
			return cudaStatus;
		}
		if((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed at clusterData\n");
			free(d_labelChanges);
			return cudaStatus;
		}
		//assigns data to clusters and counts sum
		clusterData<<<blockNum, BLOCK_SIZE>>>(d_data, d_sum, d_counter, d_cluster, clusterSize, dataSize, d_label, d_labelChanges);
		if((cudaStatus = cudaGetLastError()) != cudaSuccess) {
			fprintf(stderr, "clusterData failed\n");
			free(d_labelChanges);
			return cudaStatus;
		}
		if((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed at clusterData\n");
			free(d_labelChanges);
			return cudaStatus;
		}
		try {
			//gets total label changes
			totalChanges = thrust::reduce(thrust::device, thr_labelChanges, thr_labelChanges + dataSize);
		} catch(thrust::system_error ex) {
			fprintf(stderr, "thrust adjacent_difference error: %s\n", ex.what());
			free(d_labelChanges);
			return cudaStatus;
		}

		if((cudaStatus = stopTimer(&start, &stop, "clusterData", &totalTime)) != cudaSuccess) { 
			free(d_labelChanges); 
			return cudaStatus; 
		}
		//ends algorithm if there was small enough number of changes
		if(totalChanges / (double)dataSize < threshold) {
			printf("Delta: %f\n", totalChanges / (double)dataSize);
			stopped = true;
			break;
		}
		if((cudaStatus = startTimer(&start, &stop)) != cudaSuccess) { 
			free(d_labelChanges);
			return cudaStatus; 
		}
		//recalculates clusters and clears sum and counter
		refreshClusters<<<1, CLUSTER_NUM>>>(d_sum, d_cluster, d_counter);
		if((cudaStatus = cudaGetLastError()) != cudaSuccess) {
			fprintf(stderr, "refreshClusters failed\n");
			free(d_labelChanges);
			return cudaStatus;
		}
		if((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed at refreshClusters\n");
			free(d_labelChanges);
			return cudaStatus;
		}

		if((cudaStatus = stopTimer(&start, &stop, "refreshClusters", &totalTime)) != cudaSuccess) {
			free(d_labelChanges);
			return cudaStatus; 
		}
		if((cudaStatus = startTimer(&start, &stop)) != cudaSuccess) {
			free(d_labelChanges);
			return cudaStatus; 
		}
		i++;
	}
	if(!stopped) {
		if((cudaStatus = stopTimer(&start, &stop, "", &totalTime)) != cudaSuccess) {
			free(d_labelChanges);
			return cudaStatus; 
		}
	}
	printf("Total time: %f\n", totalTime);
	result->neededIterations = i;
	result->gpuTime = totalTime;
	result->delta = totalChanges / (double)dataSize;
	cudaFree(d_labelChanges);
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t kmeansInit(dim3 *data, dim3 *cluster, int *label, double threshold, long dataSize, int clusterSize, 
					   int iterations, Result *result, int blockNum)
{
    dim3 *d_data = 0, *d_sum = 0, *d_cluster = 0;
	int *d_counter = 0, *d_label = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    if(handleError(cudaStatus = cudaSetDevice(0), "cudaSetDevice failed")) {
		goto Error;
	}
	//memory allocation
	if(handleError(cudaStatus = cudaMalloc((void**)&d_data, dataSize * sizeof(dim3)), "cudaMalloc failed at d_data")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMalloc((void**)&d_label, dataSize * sizeof(int)), "cudaMalloc failed at d_label")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMalloc((void**)&d_sum, clusterSize * sizeof(dim3)), "cudaMalloc failed at d_sum")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMalloc((void**)&d_cluster, clusterSize * sizeof(dim3)), "cudaMalloc failed at d_cluster")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMalloc((void**)&d_counter, clusterSize * sizeof(int)), "cudaMalloc failed at d_counter")) {
		goto Error;
	}
	//memory set
	//if(handleError(cudaStatus = cudaMemset(&d_label, -1, dataSize * sizeof(int)), "cudaMemset failed at d_label")) {
	//	goto Error;
	//}
	if(handleError(cudaStatus = cudaMemset(d_sum, 0, clusterSize * sizeof(dim3)), "cudaMemset failed at d_sum")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMemset(d_counter, 0, clusterSize * sizeof(int)), "cudaMemset failed at d_counter")) {
		goto Error;
	}
	//memory copy
	if(handleError(cudaStatus = cudaMemcpy(d_data, data, dataSize * sizeof(dim3), cudaMemcpyHostToDevice), "cudaMemcpy failed at d_data")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMemcpy(d_cluster, cluster, clusterSize * sizeof(dim3), cudaMemcpyHostToDevice), "cudaMemcpy failed at d_cluster")) {
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
	if(handleError(cudaStatus = kmeans(d_data, d_cluster, d_label, d_counter, d_sum, threshold, dataSize, clusterSize, iterations, result, blockNum), "kmeans failed")) {
		goto Error;
	}

	//copy results back
	if(handleError(cudaStatus = cudaMemcpy(label, d_label, dataSize * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed at label")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMemcpy(cluster, d_cluster, clusterSize * sizeof(dim3), cudaMemcpyDeviceToHost), "cudaMemcpy failed at cluster")) {
		goto Error;
	}

Error:
	cudaFree(d_data);
	cudaFree(d_sum);
	cudaFree(d_cluster);
	cudaFree(d_label);
	cudaFree(d_counter);
    
    return cudaStatus;
}
