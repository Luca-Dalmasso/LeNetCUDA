/**
 * @file common.cuh
 * @brief library for common used functions and macros in a CUDA applications
 */

#ifndef _COMMON
#define _COMMON

/**
* @defgroup set of macros for this application (default all to 0)
* @{
*/

/** enable verbose stdout (disable this when profiling)*/
#define VERBOSE 0

/** shared memory padding size (0= no padding, 1= used for 4byte banks, 2=used when shared memory has 8byte banks)*/
#define IPAD 0

/** enable host computations for error checking*/
#define CHECK 1

/** maximum difference allowed between a GPU and a CPU result in order to consider them equal (used for fast math intrinsic functions)*/
#define DELTA 0.000001f

/** PTX Unsigned Integer __umul24 instruction*/
#define UIMUL24(a,b) (__umul24((a),(b)))

/** PTX Signed Integer __mul24 instruction*/
#define IMUL24(a,b) (__mul24((a),(b))) 

/** PTX Unsigned Integer __umad24 instruction*/
#define UIMAD24(a,b,c) (__umad24((a),(b),(c)))

/** PTX Signed Integer __mad24 instruction*/
#define IMAD24(a,b,c) (__mad24((a),(b),(c)))

/** @} */

typedef unsigned char uint8_t;

/**
 * @brief check if the cuda call correctly worked
 * example: CHECK_CUDA(cudaMalloc(..));
 */
#define CHECK_CUDA(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(-1);															   \
    }                                                                          \
}



/**
 * @brief check pointer validity
 */
#define CHECK_PTR(ptr)                                                          \
{                                                                              	\
    if (ptr == NULL)                                                  			\
    {                                                                          	\
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                 	\
        fprintf(stderr, "Null pointer\n" );				                        \
		exit(-1);						       									\
    }                                                                          	\
    																			\
}

/**
 * @brief function that returns a random number in range [0-255]
 */
uint8_t randomUint8(void);

/**
 * @brief function that returns time in seconds using gettimeofday system call to get system's clock
 * usage: tStart=cpuSecond(); [some computations] tElapsed=cpuSecond()-tStart;
 */
double cpuSecond(void);

/**
 * @Brief function used to check if gpu and cpu results are the same
 * @param host: host result in form of array or matrix
 * @param device: device result in form of array o matrix
 * @param nx: array's size X
 * @param ny: number of rows of the matrix (PUT 1 if 1D array)
 * @return 0 if equals, 1 if NOT
 */
uint8_t checkRes(float *host, float *device, int nx, int ny);

/**
 * @brief query info from your GPU
 */
void initCUDA(void);


#endif















