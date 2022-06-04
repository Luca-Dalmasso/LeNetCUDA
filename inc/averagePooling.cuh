/**
 * @file averagePooling.h
 * @brief library for average pooling algorithm containing both Host and Device prototypes
 */

#ifndef _AVGPOOL_
#define _AVGPOOL_

/**Stride (leave this fixed)*/
#define STRIDE 2
/**Pool Size (leave this fixed)*/
#define POOLSIZE 2


/**
 * @brief host (CPU) function to compute average pooling (row major)
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @warning this algorithm works only with fixed STRIDE and POOLSIZE!
 */
void hostAvgPool(float *in, float *out, int nx, int ny);


/**
 * @brief naive version of average pooling algorithm on device
 * every thread load an element from source matrix and compute the division by 4
 * then saves the result in a shared memory of [blockDimx][blockDimy] size.
 * then all even threads in a block will load 4 elements and compute the final sum of the average pool.
 * finally the result will be loaded in global 
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @warning this algorithm works only with the fixed STRIDE and POOLSIZE!
 * @warning this algorithm works only for matrixes and blocks that are divisible by 2 
 */
__global__ void naiveAvgPool(float *in, float *out, int nx, int ny);


/**
 * @brief naive version without shared memory
 * @param in: source matrix
 * @param out:  output matrix
 * @param nx: matrix's columns
 * @param ny: matrix's rows
 * @warning this algorithm works only with the fixed STRIDE and POOLSIZE!
 * @warning this algorithm works only for matrixes and blocks that are divisible by 2 
 */
__global__ void naiveAvgPoolV2(float *in, float *out, int nx, int ny);

#endif
