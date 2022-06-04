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
/*kernel size is the unrolling factor for loop unrolling and default size of the filters*/
#define KERNEL_SIZE 5 

/** @} */


/**
 * @brief host (CPU) function used to perform convolution layer
 * each output element out[i][j]=sigmoid(bias+convolution)
 * ouput matrix size X=nx-kernelSize+1, Y=ny-kernelSize+1
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @param bias
 */
void hostConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);

/**
 * @brief naive version of convolution layer on device
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @param bias
 */
__global__ void naiveConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);


/**
 * @brief naive version of convolution layer on device with loop unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @param bias
 * @warning KERNEL_SIZE is the unrolling factor, rewrite this fucntion for different unrolling factors
 */
__global__ void naiveUnrolledConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);


/**
 * @brief convolution layer on device with shared memory tiling
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @param bias
 */
__global__ void sharedConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);

/**
 * @brief convolution layer with tiled shared memory and loop unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @warning KERNEL_SIZE is the unrolling factor, rewrite this fucntion for different unrolling factors
 */
__global__ void sharedUnrollConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);

/**
 * @brief convolution layer with tiled shared memory, loop unrolling and intrinsic functions
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
  * @warning KERNEL_SIZE is the unrolling factor, rewrite this fucntion for different unrolling factors
 */
__global__ void approxConvLayer(float *in, float *out, int nx, int ny, float *filter, int kernelSize, float bias);



#endif
