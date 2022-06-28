/**
 * @file LeNet.cu
 * @brief LeNet-1 Forward propagation algoritm Host+Device functions
 */

#include <stdio.h>
#include <math.h>
#include "common.cuh"
#include "LeNet.cuh"

/**
 *@brief Sigmoid activation function
 *@param a: input pixel
 */
__host__ __device__ static inline float sigmoid(float a)
{
	return 1/(1+exp(-a));
}


/**
 * @brief Host Average Pooling Algorithm, Tile 2x2, Stride 2
 * @param in: input matrix
 * @param out: output matrix
 * @param isize: input matrix size
 * @param osize: output matrix size
 */
__host__ void hostAvgPool(float *in, float *out, int isize, int osize)
{
	int i,j,k,p;
	float sum;
	int oi,oj;
	for(i=0,oi=0;i<isize;oi++,i=i+2)
	{
		for(j=0,oj=0;j<isize;oj++,j=j+2)
		{
			sum=0.0;
			for(k=i;k<i+2;k++)
			{
				for(p=j;p<j+2;p++)
				{
					sum+=in[k*isize +p];
				}
			}
			out[oi*osize+oj]=sum/4.0;
		}
	}
}

/**
 * @brief Host Convolution Algorithm embedded with bias and activation
 * @param in: input matrix
 * @param out: output matrix
 * @param filter: convolution filter
 * @param fsize: convolution filter size
 * @param isize: input matrix size
 * @param osize: output matrix size
 */
__host__ void hostConvolveActive(float *in, float *out, float *filter, int fsize, int isize, int osize)
{
	float sum = 0;
   	int center = (fsize>>1); //filter is centered on the pixels
   	int ii, jj;
   	for (int i = center, oi=0; i<(isize-center);oi++, i++) //borders are not considered
   	{ 
     	for (int j = center, oj=0; j<(isize-center);oj++, j++) //borders are not considered
     	{
       		sum = 0;
       		for (int ki = 0; ki<fsize; ki++)
       		{
 				for (int kj = 0; kj<fsize; kj++)
 				{
 	  				jj = kj + j - center;
 	  				ii = ki + i - center;
 	  				sum+=in[ii*isize+jj]*filter[ki*fsize + kj];
 				}
       			out[oi*osize+oj] = sigmoid(sum + BIAS);
     		}
     	}
     }
}


/**
 * @brief create and fill fill filters with random values
 * @return Struct Weights type 
 */
__host__ static Weigths* initFilters()
{
	Weigths *weights = (Weigths *)malloc(sizeof(struct Weigths));
	CHECK_PTR(weights);
	int i,j;
	for(i=0;i<C1;i++)
		for(j=0; j<LENGTH_KERNEL0*LENGTH_KERNEL0; j++)	//C1
			weights->filters1[i][j]=randomUint8()/150.0f;
	for(i=0;i<C2;i++)
		for(j=0; j<LENGTH_KERNEL0*LENGTH_KERNEL0; j++)  //C2
			weights->filters2[i][j]=randomUint8()/150.0f;
	for(i=0;i<C3;i++)
		for(j=0; j<LENGTH_KERNEL1*LENGTH_KERNEL1; j++)  //C3
			weights->filters3[i][j]=randomUint8()/150.0f;
	return weights; 
}

/**
 * @brief fill Feature0 (input image) with random values
 * @param image: input image 28x28
 */
__host__ static void initImage(float image[LENGTH_FEATURE0*LENGTH_FEATURE0])
{
	int i;
	for(i=0;i<LENGTH_FEATURE0*LENGTH_FEATURE0;i++)
		image[i]=randomUint8()/16.0f;
}

/*
 * @brief init Cluster struct as a collection of images to be classified in parallel
 */
__host__ static Cluster* initCluster()
{
	Cluster *c = (Cluster *)malloc(sizeof(struct Cluster));
	CHECK_PTR(c);
	for(int i=0;i<NBLOCKS;i++)
		initImage(c->image_collection[i]);
	return c;
}

/**
 * @brief create Feature Struct type
 * @return Feature Struct type
 */
__host__ static Feature* initFeat()
{
	Feature *feat = (Feature *)malloc(sizeof(struct Feature));
	CHECK_PTR(feat);
	return feat;
} 

/**
 * @brief copy array values into another
 * @param s: source array
 * @param d: destination array
 * @param dim: size of the copy
 */
__host__ static void arrcpy(float *s, float *d, int dim)
{
	for(int i=0;i<dim;i++)
		d[i]=s[i];
}


/**
 * @brief Host Forward Propagation LAYER1
 * convolution layer C1: IMAGE --> C1 --> LAYER1 composed of 4 features 24x24 each
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostLayer1(Feature *feats, Weigths *weights)
{
	hostConvolveActive(feats->image, feats->layer1[0],weights->filters1[0], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
	hostConvolveActive(feats->image, feats->layer1[1],weights->filters1[1], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);	
	hostConvolveActive(feats->image, feats->layer1[2],weights->filters1[2], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
	hostConvolveActive(feats->image, feats->layer1[3],weights->filters1[3], LENGTH_KERNEL0, LENGTH_FEATURE0, LENGTH_FEATURE1);
}

/**
 * @brief Host Forward Propagation LAYER2
 * pooling layer, first downsampling layer: LAYER1-->S1-->LAYER2 composed of 4 features 12x12 each
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostLayer2(Feature *feats,  Weigths *weights)
{
	hostAvgPool(feats->layer1[0], feats->layer2[0], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[1], feats->layer2[1], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[2], feats->layer2[2], LENGTH_FEATURE1, LENGTH_FEATURE2);
	hostAvgPool(feats->layer1[3], feats->layer2[3], LENGTH_FEATURE1, LENGTH_FEATURE2);
}

/**
 * @brief Host Forward Propagation LAYER3
 * convolution layer C2: LAYER2-->C2-->LAYER3 composed of 12 features 8x8 each
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostLayer3(Feature *feats, Weigths *weights)
{
	hostConvolveActive(feats->layer2[0], feats->layer3[0], weights->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[0], feats->layer3[1], weights->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[0], feats->layer3[2], weights->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);	
	hostConvolveActive(feats->layer2[1], feats->layer3[3], weights->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[1], feats->layer3[4], weights->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[1], feats->layer3[5], weights->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);	
	hostConvolveActive(feats->layer2[2], feats->layer3[6], weights->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[2], feats->layer3[7], weights->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[2], feats->layer3[8], weights->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[9], weights->filters2[0], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[10], weights->filters2[1], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
	hostConvolveActive(feats->layer2[3], feats->layer3[11], weights->filters2[2], LENGTH_KERNEL0, LENGTH_FEATURE2, LENGTH_FEATURE3);
}

/**
 * @brief Host Forward Propagation LAYER4
 * pooling layer, second downsampling layer: LAYER3-->S2-->LAYER4 composed of 12 features 4x4 each
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostLayer4(Feature *feats, Weigths *weights)
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

/**
 * @brief Host Forward Propagation LAYER5
 * convolution layer C3: LAYER4 -->C3-->LAYER5 (OUTPUT Layer) 10 output values
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostOutputEval(Feature *feats, Weigths *weights)
{
	float partial;
	for(int j=0;j<OUTPUT;j++)
	{
		partial = 0.0;
		for(int i=0;i<LAYER4;i++)
		{		       				
       		for (int ki = 0; ki<LENGTH_KERNEL1; ki++)
       		{
 				for (int kj = 0; kj<LENGTH_KERNEL1; kj++)
 				{
 	  				partial+=(feats->layer4[i][ki*LENGTH_FEATURE4+kj]) * weights->filters3[j][ki*LENGTH_KERNEL1 + kj];
 				}
     		} 	 		
		}
		feats->layer5[j]=sigmoid(partial+BIAS);
	}
}

/**
 * @brief Host Forward Propagation
 * LAYER1 + LAYER2 + LAYER3 + LAYER4 + LAYER5
 * @param feats: feature type
 * @param weights: weights type
 */
__host__ void hostForward(Feature *feats, Weigths *weights)
{
	//Input  --> Layer1
	hostLayer1(feats, weights);
	//Layer1 --> Layer2
	hostLayer2(feats, weights);
	//Layer2 --> Layer3
	hostLayer3(feats, weights);
	//Layer3 --> Layer4
	hostLayer4(feats, weights);
	//Layer4 --> Output
	hostOutputEval(feats, weights);
}

/**
 * @brief print on file the entire LeNet data type
 * @param feats: feature type
 * @param weights: weights type
 * @param fp: file pointer
 */
void printLeNet(Feature *feats, Weigths *weights, FILE *fp)
{
	CHECK_PTR(weights);
	CHECK_PTR(feats);
	CHECK_PTR(fp);
	int i,j,k;
	fprintf(fp,"LAYER0: input image, size=[%dx%d]\n", LENGTH_FEATURE0, LENGTH_FEATURE0);
	for(i=0;i<LENGTH_FEATURE0;i++)
	{
		for(j=0; j<LENGTH_FEATURE0; j++)
		{
			fprintf(fp,"%f ",feats->image[i*LENGTH_FEATURE0+j]);	
		}
		fprintf(fp,"\n");
	}		
	fprintf(fp,"C1 %d Weights, size=[%dx%dx%d]:\n", C1, C1, LENGTH_KERNEL0, LENGTH_KERNEL0);
	for(i=0;i<C1;i++)
	{
		for(j=0; j<LENGTH_KERNEL0; j++)
		{
			for(k=0;k<LENGTH_KERNEL0;k++)
			{
				fprintf(fp,"%f ",weights->filters1[i][j*LENGTH_KERNEL0+k]);
			}	
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
	fprintf(fp,"LAYER1: %d Features, size=[%dx%dx%d]:\n",LAYER1, LAYER1, LENGTH_FEATURE1, LENGTH_FEATURE1);
	for(i=0;i<LAYER1;i++)
	{
		for(j=0;j<LENGTH_FEATURE1;j++)
		{
			for(k=0;k<LENGTH_FEATURE1;k++)
			{
				fprintf(fp,"%f ",feats->layer1[i][j*LENGTH_FEATURE1+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}		
	fprintf(fp,"LAYER2: %d Features, size=[%dx%dx%d]:\n",LAYER2, LAYER2, LENGTH_FEATURE2, LENGTH_FEATURE2);
	for(i=0;i<LAYER2;i++)
	{
		for(j=0;j<LENGTH_FEATURE2;j++)
		{
			for(k=0;k<LENGTH_FEATURE2;k++)
			{
				fprintf(fp,"%f ",feats->layer2[i][j*LENGTH_FEATURE2+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
	fprintf(fp,"C2 %d Weights, size=[%dx%dx%d]:\n", C2, C2, LENGTH_KERNEL0, LENGTH_KERNEL0);
	for(i=0;i<C2;i++)
	{
		for(j=0; j<LENGTH_KERNEL0; j++)
		{
			for(k=0;k<LENGTH_KERNEL0;k++)
			{
				fprintf(fp,"%f ",weights->filters2[i][j*LENGTH_KERNEL0+k]);	
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
	fprintf(fp,"LAYER3: %d Features, size=[%dx%dx%d]:\n",LAYER3, LAYER3, LENGTH_FEATURE3, LENGTH_FEATURE3);
	for(i=0;i<LAYER3;i++)
	{
		for(j=0;j<LENGTH_FEATURE3;j++)
		{
			for(k=0;k<LENGTH_FEATURE3;k++)
			{
				fprintf(fp,"%f ",feats->layer3[i][j*LENGTH_FEATURE3+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}	
	fprintf(fp,"LAYER4: %d Features, size=[%dx%dx%d]:\n",LAYER4, LAYER4, LENGTH_FEATURE4, LENGTH_FEATURE4);
	for(i=0;i<LAYER4;i++)
	{
		for(j=0;j<LENGTH_FEATURE4;j++)
		{
			for(k=0;k<LENGTH_FEATURE4;k++)
			{
				fprintf(fp,"%f ",feats->layer4[i][j*LENGTH_FEATURE4+k]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"C3 %d Weights, size=[%dx%dx%d]:\n", C3, C3, LENGTH_KERNEL1, LENGTH_KERNEL1);
	for(i=0;i<C3;i++)
	{
		for(j=0; j<LENGTH_KERNEL1; j++)
		{
			for(k=0;k<LENGTH_KERNEL1;k++)
			{
				fprintf(fp,"%.2f ",weights->filters3[i][j*LENGTH_KERNEL1+k]);
			}	
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"OUTPUT: %d Features, size=[%dx%dx%d]:\n",OUTPUT, OUTPUT, LENGTH_FEATURE5, LENGTH_FEATURE5);
	for(int i=0;i<OUTPUT;i++)
		fprintf(fp,"%f ", feats->layer5[i]);
	fprintf(fp,"\n");	
}

/**
 * @brief Device Average Pooling on Tile 2x2
 * @param in: input matrix
 * @param out: output matrix
 * @param isize: input matrix size
 * @param osize: output matrix size
 */
__device__  inline void deviceAvgPool(float *in, float *out, int isize, int osize)
{
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ox;
	int oy;
	float sum=0.0f;
	sum+=in[ty*isize+tx];
	sum+=in[ty*isize+tx+1];
	sum+=in[(ty+1)*isize+tx];
	sum+=in[(ty+1)*isize+tx+1];
	sum=sum/4.0f;
	ox=(tx>>1); //resize index for the output matrix 
	oy=(ty>>1);
	out[oy*osize+ox]=sum;	
}

/**filters, struct Weights unpacked in constant memory */
__constant__ float filtersC1[C1][LENGTH_KERNEL0*LENGTH_KERNEL0];
__constant__ float filtersC2[C2][LENGTH_KERNEL0*LENGTH_KERNEL0]; 
__constant__ float filtersC3[C3][LENGTH_KERNEL1*LENGTH_KERNEL1];

/**
 * @brief Device forward propagation algorithm shared + constant memory version
 * @param in: feature 0 (input image) [GLOBAL MEMORY]
 * @param out: output layer
 * @warning: this kernel contains a lot of synchronization points in the code
 * some of them are mandatory, others are used to avoid that the threads run out of resources at runtime
 * you can try to remove some of those synchronization points with care..in that case the application might crash
 * probably because at a certain point there are too many threads concurrenly active and they will use all the registers available in the SM..
 */
__global__ void deviceForwardV3(float in[LENGTH_FEATURE0*LENGTH_FEATURE0], float out[OUTPUT])
{
	//features, struct Feature unpacked in shared memory
	__shared__ float image[LENGTH_FEATURE0*LENGTH_FEATURE0];
	__shared__ float layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1];
	__shared__ float layer2[LAYER2][LENGTH_FEATURE2*LENGTH_FEATURE2];
	__shared__ float layer3[LAYER3][LENGTH_FEATURE3*LENGTH_FEATURE3];
	__shared__ float layer4[LAYER4][LENGTH_FEATURE4*LENGTH_FEATURE4];
	__shared__ float tmp_output[OUTPUT][LAYER4];
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	float finalResult;
	float sum[12];
	
	if(tx>=LENGTH_FEATURE0 || ty>=LENGTH_FEATURE0) return;
	
	//LAYER0 raw copy in shared memory
	image[ty*LENGTH_FEATURE0+tx]=in[(ty*LENGTH_FEATURE0+tx)];
	__syncthreads();
	
	//LAYER1: C1 convolutional layer
	if(tx>=CENTER && tx<LENGTH_FEATURE0-CENTER && ty>=CENTER && ty<LENGTH_FEATURE0-CENTER) //borders are not considered
	{ 
		#pragma unroll(5)
 		for(int i=0;i<LENGTH_KERNEL0;i++)
 		{
 			#pragma unroll(5)
 			for(int j=0;j<LENGTH_KERNEL0;j++)
 			{
 				sum[0]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[0][(i*LENGTH_KERNEL0)+j];
 				sum[1]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[1][(i*LENGTH_KERNEL0)+j];
 				sum[2]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[2][(i*LENGTH_KERNEL0)+j];
 				sum[3]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[3][(i*LENGTH_KERNEL0)+j];
 			}
 		}
 		layer1[0][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[0] + BIAS);
 		layer1[1][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[1] + BIAS);
 		layer1[2][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[2] + BIAS);
 		layer1[3][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[3] + BIAS);
 	}
	__syncthreads();
	
	//LAYER2: P1 pooling layer
	if (tx<LENGTH_FEATURE1 && ty<LENGTH_FEATURE1 && tx%2==0 && ty%2==0)
	{
		deviceAvgPool(layer1[0], layer2[0], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[1], layer2[1], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[2], layer2[2], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[3], layer2[3], LENGTH_FEATURE1, LENGTH_FEATURE2);
	}
	__syncthreads();
	
	//LAYER3: C2 convolutional layer
	if(tx>=CENTER && tx<LENGTH_FEATURE2-CENTER && ty>=CENTER && ty<LENGTH_FEATURE2-CENTER) //borders are not considered
	{ 
		#pragma unroll(12)
		for(int i=0;i<12;i++)
			sum[i]=0.0f;
		#pragma unroll(5)
 		for(int i=0;i<LENGTH_KERNEL0;i++)
 		{
 			#pragma unroll(5)
 			for(int j=0;j<LENGTH_KERNEL0;j++)
 			{
 				sum[0]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[1]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[2]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[3]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[4]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[5]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[6]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[7]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[8]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[9]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[10]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[11]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 			}
 		}
 		layer3[0][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[0] + BIAS);
 		layer3[1][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[1] + BIAS);
 		layer3[2][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[2] + BIAS);
 		layer3[3][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[3] + BIAS);
 		layer3[4][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[4] + BIAS);
 		layer3[5][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[5] + BIAS);
 		layer3[6][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[6] + BIAS);
 		layer3[7][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[7] + BIAS);
 		layer3[8][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[8] + BIAS);
 		layer3[9][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[9] + BIAS);
 		layer3[10][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[10] + BIAS);
 		layer3[11][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[11] + BIAS);
 	}
 	__syncthreads();
 	
	//LAYER4: P2 pooling layer
	if (tx<LENGTH_FEATURE3 && ty<LENGTH_FEATURE3 && tx%2==0 && ty%2==0)
	{
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
	}
	__syncthreads();
	
	//LAYER5: Fully connected to OUTPUT
	if(ty<LAYER4 && tx<OUTPUT)
	{
		finalResult=0.0f;
		#pragma unroll(4)
 		for(int i=0;i<LENGTH_KERNEL1;i++)
 			#pragma unroll(4)
 			for(int j=0;j<LENGTH_KERNEL1;j++)
 				finalResult+=layer4[ty][i*LENGTH_FEATURE4+j] *  filtersC3[tx][i*LENGTH_KERNEL1+j];
		tmp_output[tx][ty]=finalResult;
		__syncthreads();
		if(ty==0)
		{
			finalResult=0.0f;
			#pragma unroll(12)
			for(int i=0;i<LAYER4;i++)
				finalResult+=tmp_output[tx][i];
			out[tx]=sigmoid(finalResult+BIAS);
		}
	}
	
}


/** @brief same FP algorithm, different data structure
 *  in this version the kernel is supposed to concurrently run NBLOCKS blocks that "in parallel" perform
 *  the Forward Propagation on NBLOCKS images.
 *  Each block basically runs his own FP on his own image and his own OUTPUT
 */
__global__ void deviceForwardBlocks(Cluster *c)
{
	//features, struct Feature unpacked in shared memory
	__shared__ float image[LENGTH_FEATURE0*LENGTH_FEATURE0];
	__shared__ float layer1[LAYER1][LENGTH_FEATURE1*LENGTH_FEATURE1];
	__shared__ float layer2[LAYER2][LENGTH_FEATURE2*LENGTH_FEATURE2];
	__shared__ float layer3[LAYER3][LENGTH_FEATURE3*LENGTH_FEATURE3];
	__shared__ float layer4[LAYER4][LENGTH_FEATURE4*LENGTH_FEATURE4];
	__shared__ float tmp_output[OUTPUT][LAYER4];
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bid=blockIdx.x;
	float finalResult;
	float sum[12];
	
	if(tx>=LENGTH_FEATURE0 || ty>=LENGTH_FEATURE0) return;
	
	//LAYER0 raw copy in shared memory
	image[ty*LENGTH_FEATURE0+tx]=c->image_collection[bid][(ty*LENGTH_FEATURE0+tx)];
	__syncthreads();
	
	//LAYER1: C1 convolutional layer
	if(tx>=CENTER && tx<LENGTH_FEATURE0-CENTER && ty>=CENTER && ty<LENGTH_FEATURE0-CENTER) //borders are not considered
	{ 
		#pragma unroll(5)
 		for(int i=0;i<LENGTH_KERNEL0;i++)
 		{
 			#pragma unroll(5)
 			for(int j=0;j<LENGTH_KERNEL0;j++)
 			{
 				sum[0]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[0][(i*LENGTH_KERNEL0)+j];
 				sum[1]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[1][(i*LENGTH_KERNEL0)+j];
 				sum[2]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[2][(i*LENGTH_KERNEL0)+j];
 				sum[3]+=image[(ty-CENTER+i)*LENGTH_FEATURE0+tx-CENTER+j] * filtersC1[3][(i*LENGTH_KERNEL0)+j];
 			}
 		}
 		layer1[0][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[0] + BIAS);
 		layer1[1][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[1] + BIAS);
 		layer1[2][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[2] + BIAS);
 		layer1[3][(ty-CENTER)*LENGTH_FEATURE1+(tx-CENTER)]=sigmoid(sum[3] + BIAS);
 	}
	__syncthreads();
	
	//LAYER2: P1 pooling layer
	if (tx<LENGTH_FEATURE1 && ty<LENGTH_FEATURE1 && tx%2==0 && ty%2==0)
	{
		deviceAvgPool(layer1[0], layer2[0], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[1], layer2[1], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[2], layer2[2], LENGTH_FEATURE1, LENGTH_FEATURE2);
		deviceAvgPool(layer1[3], layer2[3], LENGTH_FEATURE1, LENGTH_FEATURE2);
	}
	__syncthreads();
	
	//LAYER3: C2 convolutional layer
	if(tx>=CENTER && tx<LENGTH_FEATURE2-CENTER && ty>=CENTER && ty<LENGTH_FEATURE2-CENTER) //borders are not considered
	{ 
		#pragma unroll(12)
		for(int i=0;i<12;i++)
			sum[i]=0.0f;
		#pragma unroll(5)
 		for(int i=0;i<LENGTH_KERNEL0;i++)
 		{
 			#pragma unroll(5)
 			for(int j=0;j<LENGTH_KERNEL0;j++)
 			{
 				sum[0]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[1]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[2]+=layer2[0][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[3]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[4]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[5]+=layer2[1][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[6]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[7]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[8]+=layer2[2][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 				sum[9]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[0][(i*LENGTH_KERNEL0)+j];
 				sum[10]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[1][(i*LENGTH_KERNEL0)+j];
 				sum[11]+=layer2[3][(ty-CENTER+i)*LENGTH_FEATURE2+tx-CENTER+j] * filtersC2[2][(i*LENGTH_KERNEL0)+j];
 			}
 		}
 		layer3[0][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[0] + BIAS);
 		layer3[1][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[1] + BIAS);
 		layer3[2][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[2] + BIAS);
 		layer3[3][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[3] + BIAS);
 		layer3[4][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[4] + BIAS);
 		layer3[5][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[5] + BIAS);
 		layer3[6][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[6] + BIAS);
 		layer3[7][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[7] + BIAS);
 		layer3[8][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[8] + BIAS);
 		layer3[9][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[9] + BIAS);
 		layer3[10][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[10] + BIAS);
 		layer3[11][(ty-CENTER)*LENGTH_FEATURE3+(tx-CENTER)]=sigmoid(sum[11] + BIAS);
 	}
 	__syncthreads();
 	
	//LAYER4: P2 pooling layer
	if (tx<LENGTH_FEATURE3 && ty<LENGTH_FEATURE3 && tx%2==0 && ty%2==0)
	{
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
	}
	__syncthreads();
	
	//LAYER5: Fully connected to OUTPUT
	if(ty<LAYER4 && tx<OUTPUT)
	{
		finalResult=0.0f;
		#pragma unroll(4)
 		for(int i=0;i<LENGTH_KERNEL1;i++)
 			#pragma unroll(4)
 			for(int j=0;j<LENGTH_KERNEL1;j++)
 				finalResult+=layer4[ty][i*LENGTH_FEATURE4+j] *  filtersC3[tx][i*LENGTH_KERNEL1+j];
		tmp_output[tx][ty]=finalResult;
		__syncthreads();
		if(ty==0)
		{
			finalResult=0.0f;
			#pragma unroll(12)
			for(int i=0;i<LAYER4;i++)
				finalResult+=tmp_output[tx][i];
			c->class_collection[bid][tx]=sigmoid(finalResult+BIAS);
		}
	}
	
}


/**MAIN*/

#if (PROFILE_PARALLEL_BLOCKS == 0)

	/*standard forward propagation, one image at a time*/
	int main()
	{
		initCUDA();
		
		Feature *feats;
		Weigths *weights;
		Cluster *cluster;
		weights = initFilters();
		feats = initFeat();
	
		FILE *RES,*PER;
	
		RES=fopen("./logs/running-results.txt","w");
		CHECK_PTR(RES);
		PER=fopen("./logs/running-performances.txt","w");
		CHECK_PTR(PER);
	
		double timesCPU[FORWARD_CYCLES+1];
		double timesGPU[FORWARD_CYCLES+1];
		double totTimeCPU=0.0;
		double totTimeGPU=0.0;
	
		//alloc datas on device  (float in[LENGTH_FEATURE0*LENGTH_FEATURE0], float out[OUTPUT])
		float *dSource;
		float *dDest;
		float *gpuRes;
		dim3 block (LENGTH_FEATURE0, LENGTH_FEATURE0);
		gpuRes=(float *)malloc(OUTPUT*sizeof(float));
		CHECK_PTR(gpuRes);
		CHECK_CUDA(cudaMalloc( (void**)&dDest, OUTPUT*sizeof(float)));
		CHECK_CUDA(cudaMalloc( (void**)&dSource, LENGTH_FEATURE0*LENGTH_FEATURE0*sizeof(float)));
	
		//alloc device constant memory
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC1, weights->filters1, sizeof(filtersC1)));
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC2, weights->filters2, sizeof(filtersC2)));
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC3, weights->filters3, sizeof(filtersC3)));
	
		fprintf(PER,"CPU time (s),GPU time (s), Cycle\n");
		for(int i=0; i<FORWARD_CYCLES; i++)
		{
			// host forward propagation
			initImage(feats->image); //Get new image
			timesCPU[i]=cpuSecond();
			hostForward(feats, weights);
			timesCPU[i]=cpuSecond()-timesCPU[i];
			totTimeCPU+=timesCPU[i];
		
	    	// device forward propagation
			timesGPU[i]=cpuSecond();
			cudaMemcpy(dSource, feats->image, LENGTH_FEATURE0*LENGTH_FEATURE0*sizeof(float), cudaMemcpyHostToDevice);
    		deviceForwardV3<<<1,block>>>(dSource,dDest);
			cudaDeviceSynchronize();
			//CHECK_CUDA(cudaGetLastError());
			//CHECK_CUDA(cudaDeviceSynchronize());
			cudaMemcpy(gpuRes, dDest, OUTPUT*sizeof(float), cudaMemcpyDeviceToHost);
   			timesGPU[i]=cpuSecond()-timesGPU[i];
   			totTimeGPU+=timesGPU[i];
   			if(checkRes(feats->layer5,gpuRes,OUTPUT,1)==1)
   			{
    			fprintf(stderr,"GPU and CPU result missmatch in the %d cycle\n",i);
       			exit(1);
    		}
			fprintf(PER,"%f,%f,%d\n",timesCPU[i],timesGPU[i],i);
		}
		fprintf(stdout,"\n");
		
		fprintf(stdout,"CPU required time for %d cycles of forward propagation is %f (s)\n",FORWARD_CYCLES,totTimeCPU);
		fprintf(stdout,"GPU required time for %d cycles of forward propagation is %f (s)\n",FORWARD_CYCLES,totTimeGPU);

		fprintf(RES,"Lenet dump for the last iteration\n");
		printLeNet(feats, weights, RES);
	
		CHECK_CUDA(cudaFree(dSource));
    	CHECK_CUDA(cudaFree(dDest));
    	free(gpuRes);
    	free(weights);
    	free(feats);
    
    	fclose(PER);
    	fclose(RES);

    	// reset device
    	CHECK_CUDA(cudaDeviceReset());
		return 0;
	}
#else
	/*block concurrent version of forward propagation, a kernel of NBLOCK blocks runs the FP on NBLOCK different image "in parallel"*/
	int main(void)
	{
		initCUDA();
		
		Feature *feats;
		Weigths *weights;
		Cluster *cluster;
		weights = initFilters();
		feats = initFeat();
		cluster = initCluster();
	
		double timesCPU[NBLOCKS+1];
		double totTimeCPU=0.0;
		
		int i,j,k;
		
		//cpu results are saved separately
		float class_collection_CPU[NBLOCKS][OUTPUT];
		
		//run cpu forward propagation on all image in the cluster serially
		for(i=0;i<NBLOCKS;i++)
		{
			//get new image from cluster
			arrcpy(cluster->image_collection[i], feats->image, LENGTH_FEATURE0* LENGTH_FEATURE0);
			timesCPU[i]=cpuSecond();
			hostForward(feats, weights);
			timesCPU[i]=cpuSecond()-timesCPU[i];
			totTimeCPU+=timesCPU[i];
			//save current classification in the cluster
			arrcpy(feats->layer5, class_collection_CPU[i], OUTPUT);
		}
		
		//device datas
		Cluster *dStruct;
		dim3 block (LENGTH_FEATURE0, LENGTH_FEATURE0);
		double totTimeGPU=0.0;
		
		CHECK_CUDA(cudaMalloc( (void**)&dStruct, sizeof(struct Cluster)));
		
		cudaMemcpy(dStruct, cluster, sizeof(struct Cluster), cudaMemcpyHostToDevice);
		
		//alloc device constant memory
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC1, weights->filters1, sizeof(filtersC1)));
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC2, weights->filters2, sizeof(filtersC2)));
		CHECK_CUDA(cudaMemcpyToSymbol(filtersC3, weights->filters3, sizeof(filtersC3)));
		
		//call and measure performances of GPU kernel forward propagation on all image in the cluster IN PARALLEL
		totTimeGPU=cpuSecond();
		deviceForwardBlocks<<<NBLOCKS,block>>>(dStruct);
		cudaDeviceSynchronize();
		//CHECK_CUDA(cudaGetLastError());
		//CHECK_CUDA(cudaDeviceSynchronize());
   		totTimeGPU=cpuSecond()-totTimeGPU;
   		cudaMemcpy(cluster, dStruct, sizeof(struct Cluster), cudaMemcpyDeviceToHost);
		
		//check results
		for(i=0;i<NBLOCKS;i++)
		{
			if(checkRes(class_collection_CPU[i],cluster->class_collection[i],OUTPUT,1)==1)
   			{
    			fprintf(stderr,"GPU and CPU result missmatch in the %d BLOCK\n",i);
       			exit(1);
    		}
		}
		
		fprintf(stdout,"CPU required time for %d images classification is %f (s)\n",NBLOCKS,totTimeCPU);
		fprintf(stdout,"CPU required time for %d images classification is %f (s)\n",NBLOCKS,totTimeGPU);
		
		return 0;
		
	}

#endif

