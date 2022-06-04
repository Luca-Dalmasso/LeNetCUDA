/**
 * @file averagePooling.cu
 * @brief library for average pooling algorithm containing both Host and Device codes
 * @see ../inc/matrix2Dconvolution.h
 */

#include <stdio.h>
#include "../inc/common.cuh"
#include "../inc/averagePooling.cuh"

void hostAvgPool(float *in, float *out, int nx, int ny)
{
	int i,j,k,p;
	float sum;
	int oi,oj;
	int nox=(nx>>1);
	
	for(i=0,oi=0;i<ny;oi++,i=i+STRIDE){
		for(j=0,oj=0;j<nx;oj++,j=j+STRIDE){
			sum=0.0;
			for(k=i;k<i+POOLSIZE;k++){
				for(p=j;p<j+POOLSIZE;p++){
					sum+=in[k*nx +p];
				}
			}
			out[oi*nox+oj]=sum/4.0;
		}
	}
}

__global__ void naiveAvgPool(float *in, float *out, int nx, int ny)
{
	//dynamic shared memory
	extern __shared__ float sh_mem[];
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix = blockIdx.x * blockDim.x + tx;
	int iy = blockIdx.y * blockDim.y + ty;
	int ox,oy;
	int nox;
	if (ix >= nx|| iy >= ny) return;
	
	float element=in[iy*nx + ix];
	element = element/(POOLSIZE*POOLSIZE);
	sh_mem[ty*blockDim.x +tx]=element;
	
	__syncthreads();
	
	
	float avg=0.0;
	if (tx%STRIDE==0 && ty%STRIDE==0){
		ox=(ix>>1);
		oy=(iy>>1);
		nox=(nx >> 1);
		for(int i=ty;i<ty+POOLSIZE;i++)
			for(int j=tx; j<tx+POOLSIZE;j++)
				avg+=sh_mem[i*blockDim.x+j];
		out[oy*nox + ox]=avg;	
	}
	
}


__global__ void naiveAvgPoolV2(float *in, float *out, int nx, int ny)
{
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix = blockIdx.x * blockDim.x + tx;
	int iy = blockIdx.y * blockDim.y + ty;
	int ox,oy;
	int nox;
	float sum=0.0;
	if (ix >= nx|| iy >= ny) return;
	if (tx%STRIDE==0 && ty%STRIDE==0){
		ox=(ix>>1);
		oy=(iy>>1);
		nox=(nx >> 1);
		sum=in[iy*nx+ix];
		sum+=in[iy*nx+ix+1];
		sum+=in[(iy+1)*nx+ix];
		sum+=in[(iy+1)*nx+ix+1];
		out[oy*nox + ox]=sum/4.0;	
	}
}


