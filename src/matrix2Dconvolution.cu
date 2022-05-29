/**
 * @file matrix2Dconvolution.cu
 * @brief common functions and benchmarks templates for matrix operation
 * @see ../inc/matrix2Dconvolution.h
 */
 
#include "../inc/common.h"
#include "../inc/matrix2Dconvolution.h"

void host2Dconvolution(float *in, float *out, uint_32 nx, uint_32 ny, float *filter, uint_32 kernelSize)
{
	float sum = 0;
  	int center = (kernelSize -1)/2;
  	int ii, jj;
  
  	for (int i = center; i<(ny-center); i++){
    	for (int j = center; j<(nx-center); j++){
      		sum = 0;
      		for (int ki = 0; ki<kernelSize; ki++){
				for (int kj = 0; kj<kernelSize; kj++){
	  				jj = kj + j - center;
	  				ii = ki + i - center;
	  				sum+=in[ii*nx+jj]*filter[ki*kernelSize + kj];
				}
      			out[i*nx +j] = sum;
    		}
    	}
    }
}

__global__ void naive2Dconvolution(float *in, float *out, uint_32 nx, uint_32 ny, float *filter, uint_32 kernelSize)
{
	float sum = 0;
	uint_32 center = (kernelSize -1)/2;
	uint_32 ix=blockDim.x * blockIdx.x + threadIdx.x;
	uint_32 iy=blockDim.y * blockIdx.y + threadIdx.y;
	ix = ix + center;
	iy = iy + center;
	uint_32 ii;
	uint_32 jj;
	if (ix >= (nx-center) || iy>= (ny-center)) return;
	for (uint_32 ki=0; ki<kernelSize; ki++){
		for (uint_32 kj=0; kj<kernelSize; kj++){
			jj = kj + iy - center;
	  		ii = ki + ix - center;
	  		sum+=in[ii*nx+jj]*filter[ki*kernelSize + kj];
		}
	}
	out[ix*nx + iy] = sum;
}






