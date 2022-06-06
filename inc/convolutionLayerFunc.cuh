/**
 * @file convolutionLayerFunc.cuh
 * @brief library for convolution layer, both Host and Device prototypes
 */

#ifndef _CONV2_
#define _CONV2_

/**
* @defgroup set of definitions used by convolution kernels
* @{
*/
/*Filter 5x5*/
#define FILTER5 5 
/*Filter 3x3*/
#define FILTER3 3
/*Image size 28x28*/
#define NXY 28
/*Bias*/
#define BIAS 0.0f
/** @} */


/**
 * @brief host (CPU) function used to perform convolution layer
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter
 * @param fsize: filter size
 */
void hostConvLayer(float *in, float *out,float *filter, int fsize);

/**
 * @brief naive version of convolution layer on device
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter
 * @param fsize: filter size
 */
__global__ void naiveConvLayer(float *in, float *out,float *filter, int fsize);


/**
 * @brief convolution layer on device with shared memory tiling
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter
 * @param fsize: filter size
 */
__global__ void sharedConvLayer(float *in, float *out,float *filter, int fsize);

/**
 * @brief convolution layer with tiled shared memory and loop unrolling factor 5
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 5x5
 */
__global__ void sharedUnroll5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5]);

/**
 * @brief convolution layer with tiled shared memory and loop unrolling factor 3
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 3x3
 */
__global__ void sharedUnroll3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3]);

/**
 * @brief convolution layer with tiled shared memory, loop unrolling and intrinsic functions
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 5x5
 */
__global__ void approx5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5]);

/**
 * @brief convolution layer with tiled shared memory, loop unrolling and intrinsic functions
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 3x3
 */
__global__ void approx3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3]);


/**
 * @brief custom layer working only  with 28x28 images
 * shared memory, unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 5x5
 */
__global__ void custom5ConvLayer(float *in, float *out,float filter[FILTER5*FILTER5]);


/**
 * @brief custom layer working only  with 28x28 images
 * shared memory, unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param filter: convolution filter 5x5
 */
__global__ void custom3ConvLayer(float *in, float *out,float filter[FILTER3*FILTER3]);

#endif
