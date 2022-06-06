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
	//return a;
}

/**approximate sigmoid*/
__device__ static inline float fsigmoid(float a)
{
	return FDIV(1,(1+FEXP(-a)));
	//return a;
}

void hostConvLayer(float *in, float *out,float *filter, int fsize)
{
	int i,j,k,p;
    int p_start_j,p_start_i;
    float partial;
    
    //output matrix size
    int o_sizex=NXY - fsize +1;
    
    for(i=0;i<NXY;i++){
        for(j=0;j<NXY;j++){
            partial=0.0;
            p_start_i=i-fsize/2;
            p_start_j=j-fsize/2;
            //convolution
            for(k=0;k<fsize;k++)
                for(p=0;p<fsize;p++)
                    if(((p_start_i+k)>=0) && ((p_start_i+k)<NXY)&&((p_start_j+p)>=0)&&((p_start_j+p)<NXY))
			        	partial+=in[(p_start_i+k)*NXY+(p_start_j+p)] * filter[(k*fsize)+p];
			                       
            //add bias
            partial+=BIAS;
            //activation function
            partial=sigmoid(partial);
            //save only pixels in region of interest
            if (i>=fsize/2 && i<NXY-fsize/2 && j>=fsize/2 && j<NXY-fsize/2)
            	out[p_start_i*o_sizex + p_start_j]=partial;
        }
    }
}

__global__ void naiveConvLayer(float *in, float *out,float *filter, int fsize)
{
	float sum = 0.0;
	int center = fsize/2;
	int ix=blockDim.x * blockIdx.x + threadIdx.x;
	int iy=blockDim.y * blockIdx.y + threadIdx.y;
	int ixo=ix-center;
	int iyo=iy-center;
	int ii;
	int jj;
	//output matrix size
    int o_sizex=NXY - fsize +1;
	if (ix>=NXY || iy>=NXY) return;
	
	//convolution
    jj=ix-center;
    ii=iy-center;
    for(int k=0;k<fsize;k++)
    	for(int p=0;p<fsize;p++)
        	if(((ii+k)>=0) && ((ii+k)<NXY)&&((jj+p)>=0)&&((jj+p)<NXY))
				sum+=in[(ii+k)*NXY+(jj+p)] * filter[(k*fsize)+p];
   
   //bias
   sum+=BIAS;
   //activation sigmoid
   sum=sigmoid(sum);
   //save only pixels in region of interest
   if (iy>=center && iy<NXY-center && ix>=center && ix<NXY-center)
   		out[iyo*o_sizex + ixo]=sum;
    
}

__global__ void sharedConvLayer(float *in, float *out,float *filter, int fsize)
{
	
	extern __shared__ float tileNs[];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (fsize)/2;
	int TILE_SIZE = blockDim.x - fsize +1;
    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - center;
    int col_i = col_o - center;
    
    //output matrix size
    int o_sizex=NXY - fsize +1;

    // Load tile elements
    if(row_i >= 0 && row_i < NXY && col_i >= 0 && col_i < NXY)
        tileNs[ty*blockDim.x + tx] = in[row_i*NXY + col_i];
    else
        tileNs[ty*blockDim.x + tx] = 0.0f;

    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<fsize; y++)
            for(int x=0; x<fsize; x++)
                pValue += filter[y*fsize + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        //bias
        pValue+=BIAS;
        //sigmoid
        pValue=sigmoid(pValue);
        // only write values if you are inside region of interest
        //if(row_o < ny && col_o < nx)
        if(row_o>=center && row_o<NXY-center && col_o>=center && col_o<NXY-center)
        	out[row_i*o_sizex + col_i] = pValue;
    }
}

__global__ void sharedUnroll5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5])
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = 2;
	int TILE_SIZE = blockDim.x - FILTER5 +1;
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;
    int row_i = row_o - center;
    int col_i = col_o - center;
    int o_sizex=NXY - FILTER5 +1;
    if(row_i >= 0 && row_i < NXY && col_i >= 0 && col_i < NXY)
        tileNs[ty*blockDim.x + tx] = in[row_i*NXY + col_i];
    else
        tileNs[ty*blockDim.x + tx] = 0.0f;
    __syncthreads();
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(5)
        for(int y=0; y<FILTER5; y++)
        	#pragma unroll(5)
            for(int x=0; x<FILTER5; x++)
                pValue += filter[y*FILTER5 + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        pValue+=BIAS;
        pValue=sigmoid(pValue);
        if(row_o>=center && row_o<NXY-center && col_o>=center && col_o<NXY-center)
        	out[row_i*o_sizex + col_i] = pValue;
    }
}

__global__ void sharedUnroll3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3])
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = FILTER3/2;
	int TILE_SIZE = blockDim.x - FILTER3+1;
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;
    int row_i = row_o - center;
    int col_i = col_o - center;
    if(row_i >= 0 && row_i < NXY && col_i >= 0 && col_i < NXY)
        tileNs[ty*blockDim.x + tx] = in[row_i*NXY + col_i];
    else
        tileNs[ty*blockDim.x + tx] = 0.0f;
    __syncthreads();
    //output matrix size
    int o_sizex=NXY - FILTER3 +1;
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(3)
        for(int y=0; y<FILTER3; y++)
        	#pragma unroll(3)
            for(int x=0; x<FILTER3; x++)
                pValue += filter[y*FILTER3 + x] * tileNs[(y+ty)*blockDim.x + (x+tx)];
        pValue+=BIAS;
        pValue=sigmoid(pValue);
        if(row_o>=center && row_o<NXY-center && col_o>=center && col_o<NXY-center)
        	out[row_i*o_sizex + col_i] = pValue;
    }
}


__global__ void approx5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5])
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (FILTER5>>1);
	int TILE_SIZE = blockDim.x - FILTER5 +1;
    int row_o = ty + IMUL(blockIdx.y,TILE_SIZE);
    int col_o = tx + IMUL(blockIdx.x,TILE_SIZE);
    int row_i = row_o - center;
    int col_i = col_o - center;
    int idT=IMUL(ty,blockDim.x) + tx;
    int idI=IMUL(row_i,NXY)+col_i;
    float fid;
    float iid;
    //output matrix size
    int o_sizex=NXY - FILTER5 +1;
    if(row_i >= 0 && row_i < NXY && col_i >= 0 && col_i < NXY)
        tileNs[idT] = in[idI];
    else
        tileNs[idT] = 0.0f;
    __syncthreads();
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(5)
        for(int y=0; y<FILTER5; y++){
        	#pragma unroll(5)
            for(int x=0; x<FILTER5; x++){
            	idT=IMUL(y,FILTER5)+x;
            	fid=filter[idT];
            	idT=y+ty;
            	idI=x+tx;
            	idT=IMUL(idT,blockDim.x)+idI;
            	iid=tileNs[idT];
                pValue =fid*iid + pValue; 
            }
        }
        pValue+=BIAS;
        pValue=fsigmoid(pValue);
        if(row_o>=center && row_o<NXY-center && col_o>=center && col_o<NXY-center){
        	idT=IMUL(row_i,o_sizex)+col_i;
        	out[idT] = pValue;    	
        }
    }
}

__global__ void approx3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3])
{
	extern __shared__ float tileNs[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int center = (FILTER3>>1);
	int TILE_SIZE = blockDim.x - FILTER3 +1;
    int row_o = ty + IMUL(blockIdx.y,TILE_SIZE);
    int col_o = tx + IMUL(blockIdx.x,TILE_SIZE);
    int row_i = row_o - center;
    int col_i = col_o - center;
    int idT=IMUL(ty,blockDim.x) + tx;
    int idI=IMUL(row_i,NXY)+col_i;
    float fid;
    float iid;
    //output matrix size
    int o_sizex=NXY - FILTER3 +1;
    if(row_i >= 0 && row_i < NXY && col_i >= 0 && col_i < NXY)
        tileNs[idT] = in[idI];
    else
        tileNs[idT] = 0.0f;
    __syncthreads();
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        #pragma unroll(3)
        for(int y=0; y<FILTER3; y++){
        	#pragma unroll(3)
            for(int x=0; x<FILTER3; x++){
            	idT=IMUL(y,FILTER3)+x;
            	fid=filter[idT];
            	idT=y+ty;
            	idI=x+tx;
            	idT=IMUL(idT,blockDim.x)+idI;
            	iid=tileNs[idT];
                pValue =fid*iid + pValue; 
            }
        }
        pValue+=BIAS;
        pValue=fsigmoid(pValue);
        if(row_o>=center && row_o<NXY-center && col_o>=center && col_o<NXY-center){
        	idT=IMUL(row_i,o_sizex)+col_i;
        	out[idT] = pValue;    	
        }
    }
}

__device__ static inline float conv(float in[28][28], float *filter, int ty, int tx)
{
	float sum=0.0f;
	
		#pragma unroll(5)
		for(int i=0;i<FILTER5;i++)
			#pragma unroll(5)
			for(int j=0;j<FILTER5;j++)
				sum+=in[ty-2+i][tx-2+j] * filter[(i*FILTER5)+j];
		sum+=BIAS;
		sum=sigmoid(sum);
		
	return sum;
}

__global__ void custom5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5])
{
	__shared__ float img[NXY][NXY];
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	int o_sizex=NXY - FILTER5 +1;
	img[ty][tx]=in[ty*NXY+tx];
	__syncthreads();
	if(tx>=2 && tx<NXY-2 && ty>=2 && ty<NXY-2){
		out[(ty-2)*o_sizex+(tx-2)]=conv(img, filter, ty, tx);
	}
}

__global__ void custom3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3])
{
	__shared__ float img[NXY][NXY];
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	float sum=0.0f;
	int o_sizex=NXY - FILTER3 +1;
	img[ty][tx]=in[ty*NXY+tx];
	__syncthreads();
	if(tx>=1 && tx<NXY-1 && ty>=1 && ty<NXY-1){
		#pragma unroll(3)
		for(int i=0;i<FILTER3;i++)
			#pragma unroll(3)
			for(int j=0;j<FILTER3;j++)
				sum+=img[ty-1+i][tx-1+j] * filter[(i*FILTER3)+j];
		sum+=BIAS;
		sum=sigmoid(sum);
		out[(ty-1)*o_sizex+(tx-1)]=sum;
	}
}










