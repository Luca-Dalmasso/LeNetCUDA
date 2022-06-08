

#include <stdio.h>
#include <math.h>
#include "./inc/common.cuh"
#include "./inc/LeNet.cuh"


//sigmoid
__host__ __device__ static inline float sigmoid(float a)
{
	return 1/(1+exp(-a));
	//return a;
}
//


/**
 * @brief host convolution, pooling and activation function
 */

__host__ void hostAvgPool(float *in, float *out, int isize, int osize)
{
	int i,j,k,p;
	float sum;
	int oi,oj;
	
	for(i=0,oi=0;i<isize;oi++,i=i+2){
		for(j=0,oj=0;j<isize;oj++,j=j+2){
			sum=0.0;
			for(k=i;k<i+2;k++){
				for(p=j;p<j+2;p++){
					sum+=in[k*isize +p];
				}
			}
			out[oi*osize+oj]=sum/4.0;
		}
	}
}

__host__ void hostConvolveActive(float *in, float *out, float *filter, int fsize, int isize, int osize)
{
	float sum = 0;
   	int center = (fsize>>1);
   	int ii, jj;
   	int oi,oj;

   	for (int i = center, oi=0; i<(isize-center);oi++, i++){
     	for (int j = center, oj=0; j<(isize-center);oj++, j++){
       		sum = 0;
       		for (int ki = 0; ki<fsize; ki++){
 				for (int kj = 0; kj<fsize; kj++){
 	  				jj = kj + j - center;
 	  				ii = ki + i - center;
 	  				sum+=in[ii*isize+jj]*filter[ki*fsize + kj];
 				}
       			out[oi*osize+oj] = sigmoid(sum + BIAS);
     		}
     	}
     }
}

/** */

/**
 * @brief initialization functions
 */

__host__ static LeNet1* initFilters()
{
	LeNet1 *lenet = (LeNet1 *)malloc(sizeof(struct LeNet1));
	CHECK_PTR(lenet);
	int i,j;
	for(i=0;i<C1;i++)
		for(j=0; j<LENGTH_KERNEL0*LENGTH_KERNEL0; j++)
			lenet->filters1[i][j]=randomUint8()/150.0f;
	for(i=0;i<C2;i++)
		for(j=0; j<LENGTH_KERNEL0*LENGTH_KERNEL0; j++)
			lenet->filters2[i][j]=randomUint8()/150.0f;
	for(i=0;i<C3;i++)
		for(j=0; j<LENGTH_KERNEL1*LENGTH_KERNEL1; j++)
			lenet->filters3[i][j]=randomUint8()/150.0f;
	return lenet; 
}

__host__ static void initImage(float image[LENGTH_FEATURE0*LENGTH_FEATURE0])
{
	int i;
	for(i=0;i<LENGTH_FEATURE0*LENGTH_FEATURE0;i++)
		image[i]=randomUint8()/16.0f;
}

__host__ static Feature* initFeat()
{
	Feature *feat = (Feature *)malloc(sizeof(struct Feature));
	CHECK_PTR(feat);
	initImage(feat->image);
	return feat;
} 

/** */


/**
 * @brief Host Forward Propagation
 */

__host__ void hostLayer1(Feature *feats, LeNet1 *lenet)
{
	hostConvolveActive(feats->image, feats->layer1[0],lenet->filters1[0], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
	hostConvolveActive(feats->image, feats->layer1[1],lenet->filters1[1], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);	
	hostConvolveActive(feats->image, feats->layer1[2],lenet->filters1[2], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
	hostConvolveActive(feats->image, feats->layer1[3],lenet->filters1[3], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
}

__host__ void hostLayer2(Feature *feats, LeNet1 *lenet)
{
	hostAvgPool(feats->layer1[0], feats->layer2[0], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[1], feats->layer2[1], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[2], feats->layer2[2], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[3], feats->layer2[3], LENGTH_FEATURE1, LENGTH_FEATURE2);
}

__host__ void hostLayer3(Feature *feats, LeNet1 *lenet)
{
	hostConvolveActive(feats->layer2[0], feats->layer3[0], lenet->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[0], feats->layer3[1], lenet->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[0], feats->layer3[2], lenet->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);	
	hostConvolveActive(feats->layer2[1], feats->layer3[3], lenet->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[1], feats->layer3[4], lenet->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[1], feats->layer3[5], lenet->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);	
	hostConvolveActive(feats->layer2[2], feats->layer3[6], lenet->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[2], feats->layer3[7], lenet->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[2], feats->layer3[8], lenet->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[9], lenet->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[10], lenet->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[11], lenet->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
}

__host__ void hostLayer4(Feature *feats, LeNet1 *lenet)
{
	hostAvgPool(feats->layer3[0], feats->layer4[0], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[1], feats->layer4[1], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[2], feats->layer4[2], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[3], feats->layer4[3], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[4], feats->layer4[4], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[5], feats->layer4[5], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[6], feats->layer4[6], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[7], feats->layer4[7], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[8], feats->layer4[8], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[9], feats->layer4[9], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[10], feats->layer4[10], LENGTH_FEATURE3, LENGTH_FEATURE4);
	hostAvgPool(feats->layer3[11], feats->layer4[11], LENGTH_FEATURE3, LENGTH_FEATURE4);
}

__host__ void hostOutputEval(Feature *feats, LeNet1 *lenet)
{
	float partial;
	for(int j=0;j<OUTPUT;j++){
		partial = 0.0;
		for(int i=0;i<LAYER4;i++){		       				
       		for (int ki = 0; ki<LENGTH_KERNEL1; ki++){
 				for (int kj = 0; kj<LENGTH_KERNEL1; kj++){
 	  				partial+=(feats->layer4[i][ki*LENGTH_FEATURE4+kj]) * lenet->filters3[j][ki*LENGTH_KERNEL1 + kj];
 				}
     		} 	 		
		}
		feats->layer5[j][0]=sigmoid(partial+BIAS);
	}
}

__host__ void hostForward(Feature *feats, LeNet1 *lenet)
{
	//Input  --> Layer1
	hostLayer1(feats, lenet);
	//Layer1 --> Layer2
	hostLayer2(feats, lenet);
	//Layer2 --> Layer3
	hostLayer3(feats, lenet);
	//Layer3 --> Layer4
	hostLayer4(feats, lenet);
	//Layer4 --> Output
	hostOutputEval(feats, lenet);
}

/** */


void printLeNet(Feature *feats, LeNet1 *lenet, FILE *fp)
{
	CHECK_PTR(lenet);
	CHECK_PTR(feats);
	CHECK_PTR(fp);
	int i,j,k;
	fprintf(fp,"LAYER0: input image, size=[%dx%d]\n", LENGTH_FEATURE0, LENGTH_FEATURE0);
	for(i=0;i<LENGTH_FEATURE0;i++){
		for(j=0; j<LENGTH_FEATURE0; j++){
			fprintf(fp,"%f ",feats->image[i*LENGTH_FEATURE0+j]);	
		}
		fprintf(fp,"\n");
	}	
	
	fprintf(fp,"C1 %d Weights, size=[%dx%dx%d]:\n", C1, C1, LENGTH_KERNEL0, LENGTH_KERNEL0);
	for(i=0;i<C1;i++){
		for(j=0; j<LENGTH_KERNEL0; j++){
			for(k=0;k<LENGTH_KERNEL0;k++){
				fprintf(fp,"%f ",lenet->filters1[i][j*LENGTH_KERNEL0+k]);
			}	
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"LAYER1: %d Features, size=[%dx%dx%d]:\n",LAYER1, LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1);
	for(i=0;i<LAYER1;i++){
		for(j=0;j<LENGTH_FEATURE1;j++){
			for(k=0;k<LENGTH_FEATURE1;k++){
				fprintf(fp,"%f ",feats->layer1[i][j*LENGTH_FEATURE1+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
	
	fprintf(fp,"LAYER2: %d Features, size=[%dx%dx%d]:\n",LAYER2, LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2);
	for(i=0;i<LAYER2;i++){
		for(j=0;j<LENGTH_FEATURE2;j++){
			for(k=0;k<LENGTH_FEATURE2;k++){
				fprintf(fp,"%f ",feats->layer2[i][j*LENGTH_FEATURE2+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"C2 %d Weights, size=[%dx%dx%d]:\n", C2, C2, LENGTH_KERNEL0, LENGTH_KERNEL0);
	for(i=0;i<C2;i++){
		for(j=0; j<LENGTH_KERNEL0; j++){
			for(k=0;k<LENGTH_KERNEL0;k++){
				fprintf(fp,"%f ",lenet->filters2[i][j*LENGTH_KERNEL0+k]);	
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"LAYER3: %d Features, size=[%dx%dx%d]:\n",LAYER3, LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3);
	for(i=0;i<LAYER3;i++){
		for(j=0;j<LENGTH_FEATURE3;j++){
			for(k=0;k<LENGTH_FEATURE3;k++){
				fprintf(fp,"%f ",feats->layer3[i][j*LENGTH_FEATURE3+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"LAYER4: %d Features, size=[%dx%dx%d]:\n",LAYER4, LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4);
	for(i=0;i<LAYER4;i++){
		for(j=0;j<LENGTH_FEATURE4;j++){
			for(k=0;k<LENGTH_FEATURE4;k++){
				fprintf(fp,"%f ",feats->layer4[i][j*LENGTH_FEATURE4+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"C3 %d Weights, size=[%dx%dx%d]:\n", C3, C3, LENGTH_KERNEL1, LENGTH_KERNEL1);
	for(i=0;i<C3;i++){
		for(j=0; j<LENGTH_KERNEL1; j++){
			for(k=0;k<LENGTH_KERNEL1;k++){
				fprintf(fp,"%.2f ",lenet->filters3[i][j*LENGTH_KERNEL1+k]);
			}	
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	
	fprintf(fp,"OUTPUT: %d Features, size=[%dx%dx%d]:\n",OUTPUT, OUTPUT, LENGTH_FEATURE5, LENGTH_FEATURE5);
	for(int i=0;i<OUTPUT;i++)
		fprintf(fp,"%f ", feats->layer5[i][0]);
	fprintf(fp,"\n");
		
}


/**
 * @brief device forward propagation
 */

/*5x5 filters*/
__device__  inline void deviceConvolveActive(float *in, float *out, float *filter, int isize, int osize)
{
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	float sum=0.0f;
	if(tx>=CENTER && tx<isize-CENTER && ty>=CENTER && ty<isize-CENTER){
		#pragma unroll(5)
 		for(int i=0;i<LENGTH_KERNEL0;i++)
 			#pragma unroll(5)
 			for(int j=0;j<LENGTH_KERNEL0;j++)
 				sum+=in[(ty-CENTER+i)*isize+tx-CENTER+j] * filter[(i*LENGTH_KERNEL0)+j];
 		out[(ty-CENTER)*osize+(tx-CENTER)]=sigmoid(sum + BIAS);
 	}
}

__device__  inline void deviceAvgPool(float *in, float *out, int isize, int osize)
{
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ox;
	int oy;
	float sum=0.0f;
	if (tx<isize && ty<isize && tx%2==0 && ty%2==0){
		sum+=in[ty*isize+tx];
		sum+=in[ty*isize+tx+1];
		sum+=in[(ty+1)*isize+tx];
		sum+=in[(ty+1)*isize+tx+1];
		sum=sum/4.0f;
		ox=(tx>>1);
		oy=(ty>>1);
		out[oy*osize+ox]=sum;
	}	
}

__device__ inline float miniConv(float *in, float *filter)
{
	float sum=0.0f;
	#pragma unroll(5)
 	for(int i=0;i<LENGTH_KERNEL1;i++)
 		#pragma unroll(5)
 		for(int j=0;j<LENGTH_KERNEL1;j++)
 			sum+=in[i*LENGTH_FEATURE4+j] * filter[i*LENGTH_KERNEL1+j];
 	return sum;
}

//filters (struct lenet) in constant memory
__constant__ float filtersC1[C1][LENGTH_KERNEL0*LENGTH_KERNEL0];
__constant__ float filtersC2[C2][LENGTH_KERNEL0*LENGTH_KERNEL0]; 
__constant__ float filtersC3[C3][LENGTH_KERNEL1*LENGTH_KERNEL1];

__global__ void deviceForwardV1(float in[LENGTH_FEATURE0*LENGTH_FEATURE0], float out[OUTPUT])
{
	//Features in shared memory
	__shared__ float image[LENGTH_FEATURE0*LENGTH_FEATURE0];
	__shared__ float layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1];
	__shared__ float layer2[LAYER2][LENGTH_FEATURE2*LENGTH_FEATURE2];
	__shared__ float layer3[LAYER3][LENGTH_FEATURE3*LENGTH_FEATURE3];
	__shared__ float layer4[LAYER4][LENGTH_FEATURE4*LENGTH_FEATURE4];
	__shared__ float tmp_output[OUTPUT][LAYER4];
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	float finalResult=0.0f;
	
	if(tx>=LENGTH_FEATURE0 || ty>=LENGTH_FEATURE0) return;
	//save image in shared
	image[ty*LENGTH_FEATURE0+tx]=in[ty*LENGTH_FEATURE0+tx];
	__syncthreads();
	//LAYER1
	deviceConvolveActive(image, layer1[0], filtersC1[0], LENGTH_FEATURE0, LENGTH_FEATURE1);	
	__syncthreads();
	deviceConvolveActive(image, layer1[1], filtersC1[1], LENGTH_FEATURE0, LENGTH_FEATURE1);
	__syncthreads();
	deviceConvolveActive(image, layer1[2], filtersC1[2], LENGTH_FEATURE0, LENGTH_FEATURE1);
	__syncthreads();
	deviceConvolveActive(image, layer1[3], filtersC1[3], LENGTH_FEATURE0, LENGTH_FEATURE1);
	__syncthreads();
	//LAYER2
	deviceAvgPool(layer1[0], layer2[0], LENGTH_FEATURE1, LENGTH_FEATURE2);
	deviceAvgPool(layer1[1], layer2[1], LENGTH_FEATURE1, LENGTH_FEATURE2);
	deviceAvgPool(layer1[2], layer2[2], LENGTH_FEATURE1, LENGTH_FEATURE2);
	deviceAvgPool(layer1[3], layer2[3], LENGTH_FEATURE1, LENGTH_FEATURE2);
	__syncthreads();
	//LAYER3
	deviceConvolveActive(layer2[0], layer3[0], filtersC2[0], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[0], layer3[1], filtersC2[1], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[0], layer3[2], filtersC2[2], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[1], layer3[3], filtersC2[0], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[1], layer3[4], filtersC2[1], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[1], layer3[5], filtersC2[2], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[2], layer3[6], filtersC2[0], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[2], layer3[7], filtersC2[1], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[2], layer3[8], filtersC2[2], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[3], layer3[9], filtersC2[0], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[3], layer3[10], filtersC2[1], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	deviceConvolveActive(layer2[3], layer3[11], filtersC2[2], LENGTH_FEATURE2, LENGTH_FEATURE3);
	__syncthreads();
	//LAYER4
	deviceAvgPool(layer3[0], layer4[0], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[1], layer4[1], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[2], layer4[2], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[3], layer4[3], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[4], layer4[4], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[5], layer4[5], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[6], layer4[6], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[7], layer4[7], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[8], layer4[8], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[9], layer4[9], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[10], layer4[10], LENGTH_FEATURE3, LENGTH_FEATURE4);
	deviceAvgPool(layer3[11], layer4[11], LENGTH_FEATURE3, LENGTH_FEATURE4);
	__syncthreads();
	//OUTPUT
	if(tx<LAYER4 && ty<OUTPUT)
		tmp_output[ty][tx]=miniConv(layer4[tx], filtersC3[ty]);
	__syncthreads();
	if(tx==0 && ty<OUTPUT){
		for(int i=0;i<LAYER4;i++)
			finalResult+=tmp_output[ty][i];
		out[ty]=sigmoid(finalResult+BIAS);	
	}
}

/** */


int main()
{
	initCUDA();
	
	Feature *feats;
	LeNet1  *lenet;
	lenet = initFilters();
	feats = initFeat();
	
	// host forward propagation
	double timeCPU=cpuSecond();
	hostForward(feats, lenet);
	timeCPU=cpuSecond()-timeCPU;
	fprintf(stdout,"CPU time = %f\n",timeCPU);
	
	//alloc datas on device  (float in[LENGTH_FEATURE0*LENGTH_FEATURE0], float out[OUTPUT])
	float *dSource;
	float *dDest;
	float *gpuRes;
	double timeGPU;
	dim3 block (LENGTH_FEATURE0, LENGTH_FEATURE0);
	
	//layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1]
	
	gpuRes=(float *)malloc(LENGTH_FEATURE2*LENGTH_FEATURE2*sizeof(float));
	//CHECK_PTR(gpuRes);
	CHECK_CUDA(cudaMalloc( (void**)&dDest, OUTPUT*sizeof(float)));
	CHECK_CUDA(cudaMalloc( (void**)&dSource, LENGTH_FEATURE0*LENGTH_FEATURE0*sizeof(float)));
	
	//copy data on device
	CHECK_CUDA(cudaMemcpy(dSource, feats->image, LENGTH_FEATURE0*LENGTH_FEATURE0*sizeof(float), cudaMemcpyHostToDevice));
	
	//alloc device constant memory
	cudaMemcpyToSymbol(filtersC1, lenet->filters1, sizeof(filtersC1));
	cudaMemcpyToSymbol(filtersC2, lenet->filters2, sizeof(filtersC2));
	cudaMemcpyToSymbol(filtersC3, lenet->filters3, sizeof(filtersC3));
	
	// device forward propagation
	timeGPU = cpuSecond();
    deviceForwardV1<<<1,block>>>(dSource,dDest);
    CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
   	timeGPU = cpuSecond() - timeGPU;
	fprintf(stdout,"GPU time = %f\n",timeGPU);
	
	CHECK_CUDA(cudaMemcpy(gpuRes, dDest, OUTPUT*sizeof(float), cudaMemcpyDeviceToHost));
	
	if(checkRes(feats->layer5[0],gpuRes,OUTPUT,1)==1){
    	fprintf(stderr,"GPU and CPU result missmatch!\n");
       	exit(1);
    }
	
	/*
	#if (VERBOSE)
	    printLeNet(feats, lenet, stdout);
	#endif
	*/
	CHECK_CUDA(cudaFree(dSource));
    CHECK_CUDA(cudaFree(dDest));
    free(gpuRes);
    free(lenet);
    free(feats);

    // reset device
    CHECK_CUDA(cudaDeviceReset());
	return 0;
}













