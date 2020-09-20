#include <cuda.h>

#define TILE_SIZE 4
#define WINDOW_SIZE 3

// Code taken from https://github.com/detel/Median-Filtering-GPU/blob/master/medianFilter.cu and adapted from 2D to 3D

__device__ int calculateIndex(int i, int j, int k, int dim_x, int dim_y, int dim_z, int bs) {
	int dim_x2 = (int) ceil( (float)dim_x / (float)bs )*bs;
	int dim_y2 = (int) ceil( (float)dim_y / (float)bs )*bs;
	int dim_z2 = (int) ceil( (float)dim_z / (float)bs )*bs;

	int block_i = (int) floor( (float)i / (float)bs );
	int block_j = (int) floor( (float)j / (float)bs );
	int block_k = (int) floor( (float)k / (float)bs );

	int intra_block_i = i - block_i*bs;
	int intra_block_j = j - block_j*bs;
	int intra_block_k = k - block_k*bs;

	int x = bs*bs*bs*block_i + intra_block_i;
	int y = dim_x2*bs*bs*block_j + bs*intra_block_j;
	int z = dim_x2*dim_y2*bs*block_k + bs*bs*intra_block_k;

	return x + y + z;
}

extern "C" __global__ void medianFilterKernel(float *inputImage, float *outputImage, int dim_x, int dim_y, int dim_z) {
	// Set row and colum for thread.
	int x_i = blockIdx.x * blockDim.x + threadIdx.x;
	int y_i = blockIdx.y * blockDim.y + threadIdx.y;
	int z_i = blockIdx.z * blockDim.z + threadIdx.z;

	float filterVector[27] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	if((x_i==0) || (y_i==0) || (z_i==0) || (x_i>=dim_x-1) || (y_i>=dim_y-1) || (z_i>=dim_z-1)) {
		int index = calculateIndex(x_i, y_i, z_i , dim_x, dim_y, dim_z, 4);
		outputImage[index] = 0;
	}
	else {
		for (int x = 0; x < WINDOW_SIZE; x++) {
			for (int y = 0; y < WINDOW_SIZE; y++){
				for (int z = 0; z < WINDOW_SIZE; z++){
					int index = calculateIndex(x_i+x-1, y_i+y-1, z_i+z-1 , dim_x, dim_y, dim_z, 4);
					filterVector[x*WINDOW_SIZE*WINDOW_SIZE + y*WINDOW_SIZE + z] = inputImage[index];
				}
			}
		}
		for (int i = 0; i < 27; i++) {
			for (int j = i + 1; j < 27; j++) {
				if (filterVector[i] > filterVector[j]) {
					//Swap the variables.
					float tmp = filterVector[i];
					filterVector[i] = filterVector[j];
					filterVector[j] = tmp;
				}
			}
		}
		int index = calculateIndex(x_i, y_i, z_i , dim_x, dim_y, dim_z, 4);
		outputImage[index] = filterVector[13];
	}
}
