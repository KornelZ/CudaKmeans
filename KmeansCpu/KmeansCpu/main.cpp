#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#define CLUSTER_NUM 128
#define DATA_SIZE 30000000
#define THRESHOLD 0.01
typedef struct Dim3
{
	unsigned int x, y, z;
} dim3;

long findDistance(dim3 data, dim3 cluster)
{
	return (data.x - cluster.x) * (data.x - cluster.x) 
		+ (data.y - cluster.y) * (data.y - cluster.y) 
		+ (data.z - cluster.z) * (data.z - cluster.z);
}


int main()
{
	int dataSize = DATA_SIZE;
	dim3 *cluster = (dim3*)malloc(sizeof(dim3) * CLUSTER_NUM);
	if(cluster == NULL) {
		fprintf(stderr, "Malloc failed at cluster\n");
		return 1;
	}
	dim3 *data = (dim3*)malloc(sizeof(dim3) * dataSize);
	if(data == NULL) {
		fprintf(stderr, "Malloc failed at data\n");
		return 1;
	}
	dim3 *sum = (dim3*)malloc(sizeof(dim3) * CLUSTER_NUM);
	if(sum == NULL) {
		fprintf(stderr, "Malloc failed at sum\n");
		return 1;
	}
	memset(sum, 0, sizeof(dim3) * CLUSTER_NUM);
	int *label = (int*)malloc(sizeof(int) * dataSize);
	if(label == NULL) {
		fprintf(stderr, "Malloc failed at label\n");
		return 1;
	}
	memset(label, 0, sizeof(int) * dataSize);
	int *counter = (int*)malloc(sizeof(int) * CLUSTER_NUM);
	if(counter == NULL) {
		fprintf(stderr, "Malloc failed at counter\n");
		return 1;
	}
	memset(counter, 0, sizeof(int) * CLUSTER_NUM);
	srand(NULL);
	for(int i = 0; i < dataSize; i++) {
		data[i].x = rand() % 10;
		data[i].y = rand() % 10;
		data[i].z = rand() % 10;
	}

	for(int i = 0; i < CLUSTER_NUM; i++) {
		cluster[i].x = rand() % 10;
		cluster[i].y = rand() % 10;
		cluster[i].z = rand() % 10;
	}

	double threshold = THRESHOLD;
	int totalChanges = 0;
	int previousChanges = 2 * dataSize;
	int iterations = 30;
	int i = 0;
	double start = (double)clock();
	while(i < iterations)
	{
		for(int j = 0; j < dataSize; j++)
		{
			long min = LONG_MAX, dist;
			int index;
			for(int k = 0; k < CLUSTER_NUM; k++)
			{
				if(min > (dist = findDistance(data[j], cluster[k])))
				{
					min = dist;
					index = k;
				}
			}
			if(label[j] != index + 1)
			{
				label[j] = index + 1;
				totalChanges++;
			}
			sum[index].x += data[j].x;
			sum[index].y += data[j].y;
			sum[index].z += data[j].z;
			counter[index]++;
		}
		if(totalChanges / (double)dataSize < threshold)
		{
			printf("%d iterations\n", i + 1);
			printf("Delta: %f\n", (totalChanges) / (double)dataSize);
			break;
		}

		totalChanges = 0;
		for(int j = 0; j < CLUSTER_NUM; j++)
		{
			if(counter[j] != 0) {
				cluster[j].x = sum[j].x / counter[j];

				cluster[j].y = sum[j].y / counter[j];

				cluster[j].z = sum[j].z / counter[j];
			}
			else
			{
				cluster[j].x = cluster[j].y = cluster[j].z = 0;
			}


			sum[j].x = 0;
			sum[j].y = 0;
			sum[j].z = 0;
			counter[j] = 0;
		}
		i++;
	}

	double end = (double)clock();
	start /= CLOCKS_PER_SEC;
	end /= CLOCKS_PER_SEC;
	printf("Cpu runtime: %f\n", (end - start) * 1000);

	for(int j = 0; j < CLUSTER_NUM; j++)
	{
		printf("(%u %u %u)\n", cluster[j].x, cluster[j].y, cluster[j].z);
	}

	free(data);
	free(sum);
	free(label);
	free(counter);
	free(cluster);
	getchar();
	return 0;
}