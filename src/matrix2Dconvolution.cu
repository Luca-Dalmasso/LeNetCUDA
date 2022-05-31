/**
 * @file matrix2Dconvolution.cu
 * @brief common functions and benchmarks templates for matrix operation
 * @see ../inc/matrix2Dconvolution.h
 */

#include <stdio.h>
#include "../inc/common.cuh"
#include "../inc/matrix2Dconvolution.cuh"

void host2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	int i,j,k,p;
    int p_start_j,p_start_i;
    float partial;
    for(i=0;i<ny;i++){
        for(j=0;j<nx;j++){
            partial=0.0;
            p_start_i=i-kernelSize/2;
            p_start_j=j-kernelSize/2;
            for(k=0;k<kernelSize;k++){
                for(p=0;p<kernelSize;p++){
                    if(((p_start_i+k)>=0) && ((p_start_i+k)<ny)&&((p_start_j+p)>=0)&&((p_start_j+p)<nx))
			        {
			            partial+=in[(p_start_i+k)*nx+(p_start_j+p)] * filter[(k*kernelSize)+p];
			        }
                }
            }
            
            out[i*nx + j]=partial;
        }
    }
}

__global__ void naive2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	float sum = 0;
	int center = kernelSize/2;
	int ix=blockDim.x * blockIdx.x + threadIdx.x;
	int iy=blockDim.y * blockIdx.y + threadIdx.y;
	int ii;
	int jj;
	if (ix>=nx || iy>=ny) return;
	
    jj=ix-center;
    ii=iy-center;
    for(int k=0;k<kernelSize;k++){
    	for(int p=0;p<kernelSize;p++){
        	if(((ii+k)>=0) && ((ii+k)<ny)&&((jj+p)>=0)&&((jj+p)<nx))
			{
				sum+=in[(ii+k)*nx+(jj+p)] * filter[(k*kernelSize)+p];
			}
        }
    }
           
   out[iy*nx + ix]=sum;
    
}

__global__ void shared2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	
	extern __shared__ float tileNs[];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (kernelSize)/2;
	int TILE_SIZE = blockDim.x - kernelSize +1;
    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - center;
    int col_i = col_o - center;

    // Load tile elements
    if(row_i >= 0 && row_i < ny && col_i >= 0 && col_i < nx)
        tileNs[ty*blockDim.x + tx] = in[row_i*nx + col_i];
    else
        tileNs[ty*blockDim.x + tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<kernelSize; y++)
            for(int x=0; x<kernelSize; x++)
                pValue += filter[y*kernelSize + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        // only write values if you are inside matrix bounds
        if(row_o < ny && col_o < nx)
        	out[row_o*nx + col_o] = pValue;
    }
}

__global__ void naiveUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	float sum = 0;
	int center = kernelSize/2;
	int ix=blockDim.x * blockIdx.x + threadIdx.x;
	int iy=blockDim.y * blockIdx.y + threadIdx.y;
	int ii;
	int jj;
	if (ix>=nx || iy>=ny) return;
    jj=ix-center;
    ii=iy-center;
    //outer loop unrolling
    #pragma unroll(5)
    for(int k=0;k<KERNEL_SIZE;k++){
    	//inner loop unrolling
    	#pragma unroll(5)
    	for(int p=0;p<KERNEL_SIZE;p++){
        	if(((ii+k)>=0) && ((ii+k)<ny)&&((jj+p)>=0)&&((jj+p)<nx))
			{
				sum+=in[(ii+k)*nx+(jj+p)] * filter[(k*kernelSize)+p];
			}
        }
    }
           
   out[iy*nx + ix]=sum;
}

__global__ void sharedUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (kernelSize)/2;
	int TILE_SIZE = blockDim.x - kernelSize +1;
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;
    int row_i = row_o - center;
    int col_i = col_o - center;
    if(row_i >= 0 && row_i < ny && col_i >= 0 && col_i < nx)
        tileNs[ty*blockDim.x + tx] = in[row_i*nx + col_i];
    else
        tileNs[ty*blockDim.x + tx] = 0.0f;
    __syncthreads();
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(5)
        for(int y=0; y<KERNEL_SIZE; y++)
        	#pragma unroll(5)
            for(int x=0; x<KERNEL_SIZE; x++)
                pValue += filter[y*kernelSize + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        if(row_o < ny && col_o < nx)
        	out[row_o*nx + col_o] = pValue;
    }
}


__global__ void fm_sharedUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	
}










