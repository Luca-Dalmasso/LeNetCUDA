/**
 * @file convolutionLayerFunc.cu
 * @brief library for convolution layer, both Host and Device codes
 * @see ../inc/convolutionLayerFunc.h
 */

#include <stdio.h>
#include "../inc/common.cuh"
#include "../inc/convolutionLayerFunc.cuh"
#include <math.h>

/**sigmoid*/
__host__ __device__ static inline float sigmoid(float a)
{
	return 1/(1+exp(-a));
}

/**approximate sigmoid*/
__device__ static inline float fsigmoid(float a)
{
	return 1/(1+FEXP(-a));
}

void hostConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
{
	int i,j,k,p;
    int p_start_j,p_start_i;
    float partial;
    
    //output matrix size
    int o_sizex=nx - kernelSize +1;
    
    for(i=0;i<ny;i++){
        for(j=0;j<nx;j++){
            partial=0.0;
            p_start_i=i-kernelSize/2;
            p_start_j=j-kernelSize/2;
            //convolution
            for(k=0;k<kernelSize;k++)
                for(p=0;p<kernelSize;p++)
                    if(((p_start_i+k)>=0) && ((p_start_i+k)<ny)&&((p_start_j+p)>=0)&&((p_start_j+p)<nx))
			        	partial+=in[(p_start_i+k)*nx+(p_start_j+p)] * filter[(k*kernelSize)+p];
			                       
            //add bias
            partial+=bias;
            //activation function
            partial=sigmoid(partial);
            //save only pixels in region of interest
            if (i>=kernelSize/2 && i<ny-kernelSize/2 && j>=kernelSize/2 && j<nx-kernelSize/2)
            	out[p_start_i*o_sizex + p_start_j]=partial;
        }
    }
}

__global__ void naiveConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
{
	float sum = 0;
	int center = kernelSize/2;
	int ix=blockDim.x * blockIdx.x + threadIdx.x;
	int iy=blockDim.y * blockIdx.y + threadIdx.y;
	int ixo=ix-center;
	int iyo=iy-center;
	int ii;
	int jj;
	//output matrix size
    int o_sizex=nx - kernelSize +1;
	if (ix>=nx || iy>=ny) return;
	
	//convolution
    jj=ix-center;
    ii=iy-center;
    for(int k=0;k<kernelSize;k++)
    	for(int p=0;p<kernelSize;p++)
        	if(((ii+k)>=0) && ((ii+k)<ny)&&((jj+p)>=0)&&((jj+p)<nx))
				sum+=in[(ii+k)*nx+(jj+p)] * filter[(k*kernelSize)+p];
   
   //bias
   sum+=bias;
   //activation sigmoid
   sum=sigmoid(sum);
   //save only pixels in region of interest
   if (iy>=center && iy<ny-center && ix>=center && ix<nx-center)
   		out[iyo*o_sizex + ixo]=sum;
    
}

__global__ void sharedConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
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
    
    //output matrix size
    int o_sizex=nx - kernelSize +1;

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
        //bias
        pValue+=bias;
        //sigmoid
        pValue=sigmoid(pValue);
        // only write values if you are inside region of interest
        //if(row_o < ny && col_o < nx)
        if(row_o>=center && row_o<ny-center && col_o>=center && col_o<nx-center)
        	out[row_i*o_sizex + col_i] = pValue;
    }
}

__global__ void naiveUnrolledConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
{
	float sum = 0;
	int center = kernelSize/2;
	int ix=blockDim.x * blockIdx.x + threadIdx.x;
	int iy=blockDim.y * blockIdx.y + threadIdx.y;
	int ixo=ix-center;
	int iyo=iy-center;
	int ii;
	int jj;
	if (ix>=nx || iy>=ny) return;
    jj=ix-center;
    ii=iy-center;
    //output matrix size
    int o_sizex=nx - kernelSize +1;
    //conv unrolled
    #pragma unroll(5)
    for(int k=0;k<kernelSize;k++)
    	#pragma unroll(5)
    	for(int p=0;p<kernelSize;p++)
        	if(((ii+k)>=0) && ((ii+k)<ny)&&((jj+p)>=0)&&((jj+p)<nx))
				sum+=in[(ii+k)*nx+(jj+p)] * filter[(k*kernelSize)+p];
	
   sum+=bias;
   sum=sigmoid(sum);
   if (iy>=center && iy<ny-center && ix>=center && ix<nx-center)
   		out[iyo*o_sizex + ixo]=sum;
}

__global__ void sharedUnrollConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
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
    //output matrix size
    int o_sizex=nx - kernelSize +1;
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(5)
        for(int y=0; y<kernelSize; y++)
        	#pragma unroll(5)
            for(int x=0; x<kernelSize; x++)
                pValue += filter[y*kernelSize + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        pValue+=bias;
        pValue=sigmoid(pValue);
        if(row_o>=center && row_o<ny-center && col_o>=center && col_o<nx-center)
        	out[row_i*o_sizex + col_i] = pValue;
    }
}


__global__ void approxConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias)
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (kernelSize>>1);
	int TILE_SIZE = blockDim.x - kernelSize +1;
    int row_o = ty + IMUL(blockIdx.y,TILE_SIZE);
    int col_o = tx + IMUL(blockIdx.x,TILE_SIZE);
    int row_i = row_o - center;
    int col_i = col_o - center;
    int idT=IMAD(ty,blockDim.x, tx);
    int idI=IMAD(row_i,nx,col_i);
    float fid;
    float iid;
    //output matrix size
    int o_sizex=nx - kernelSize +1;
    if(row_i >= 0 && row_i < ny && col_i >= 0 && col_i < nx)
        tileNs[idT] = in[idI];
    else
        tileNs[idT] = 0.0f;
    __syncthreads();
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(5)
        for(int y=0; y<KERNEL_SIZE; y++){
        	#pragma unroll(5)
            for(int x=0; x<KERNEL_SIZE; x++){
            	idT=IMAD(y,kernelSize,x);
            	fid=filter[idT];
            	idT=y+ty;
            	idI=x+tx;
            	idT=IMAD(idT,blockDim.x,idI);
            	iid=tileNs[idT];
                pValue =fid*iid + pValue; 
            }
        }
        pValue+=bias;
        pValue=fsigmoid(pValue);
        if(row_o>=center && row_o<ny-center && col_o>=center && col_o<nx-center){
        	idT=IMAD(row_i,o_sizex,col_i);
        	out[idT] = pValue;    	
        }
    }
}










