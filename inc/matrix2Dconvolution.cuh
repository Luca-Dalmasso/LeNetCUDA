/**
 * @file matrix2Dconvolution.h
 * @brief library for 2D convolution containing both Host and Device prototypes
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
 * @brief host (CPU) function to compute 2D matrix (row major) convolution
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 */
void host2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);

/**
 * @brief naive version of 2D convolution algorithm on device
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 */
__global__ void naive2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);


/**
 * @brief naive version of 2D convolution algorithm with loop unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @warning KERNEL_SIZE is the unrolling factor
 */
__global__ void naiveUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);


/**
 * @brief 2D convolution algorithm with tiled shared memory
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 */
__global__ void shared2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);

/**
 * @brief 2D convolution algorithm with tiled shared memory and loop unrolling
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @warning KERNEL_SIZE is the unrolling factor
 */
__global__ void sharedUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);

/**
 * @brief 2D convolution algorithm with tiled shared memory, loop unrolling and intrinsic functions
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @param filter: convolution filter
 * @param kernelSize: filter's size
 * @warning KERNEL_SIZE is the unrolling factor
 */
__global__ void fm_sharedUnroll2Dconvolution(float *in, float *out, int nx, int ny, float *filter, int kernelSize);



#endif
