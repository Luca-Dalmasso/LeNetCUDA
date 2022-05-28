/**
 * @file matrix2Dconvolution.cu
 * @brief common functions and benchmarks templates for matrix operation
 * @see ../inc/matrix2Dconvolution.h
 */
 
#include "../inc/common.h"
#include "../inc/matrix2Dconvolution.h"

void host2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
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
	  				sum+=in[ii*Nx+jj]*filter[ki*kernelSize + kj];
				}
      			out[i*Nx +j] = sum;
    		}
    	}
    }
}

__global__ void naive2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize)
{
	
}
