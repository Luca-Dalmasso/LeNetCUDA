/**
 * @file matrix2Dconvolution.h
 * @brief library for 2D convolution containing both Host and Device prototypes
 */
 
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
