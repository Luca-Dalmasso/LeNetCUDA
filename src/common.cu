/**
 * @file common.cu
 * @see ../inc/common.cuh
 */
 
#include "common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double cpuSecond(void) {
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void initCUDA(void){
	int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        exit(1);
    }
    #if (VERBOSE==2)
    	fprintf(stdout,"Detected %d CUDA Capable device(s)\n", deviceCount);
    	int dev = 0, driverVersion = 0, runtimeVersion = 0;
    	CHECK_CUDA(cudaSetDevice(dev));
    	cudaDeviceProp deviceProp;
    	CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));
    	printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    	cudaDriverGetVersion(&driverVersion);
    	cudaRuntimeGetVersion(&runtimeVersion);
    	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    	printf("  Total amount of global memory:                 %.2f MBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    	printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);
    	printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    	printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);
    	if (deviceProp.l2CacheSize)
    	{
        	printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    	}
    	printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    	printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    	printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    	printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    	printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    	printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    	printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    	printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    	printf("  Maximum memory pitch:                          %lu bytes\n",
           deviceProp.memPitch);
    #endif
    cudaSetDevice(0);
}

uint8_t randomUint8(void){
	//this variable must be declared static
	static unsigned int t;
	//initialize lfsr with t as a SEED
	srand(t);
	//change the seed, the seed is declared as static so his value will remain in the memory
	t=t+13;
	return rand()%0xf;
}

uint8_t checkRes(float *host, float *device, int nx, int ny){
	int i;
	for(i=0;i<(nx*ny);i++){
		if(abs(host[i]-device[i])>DELTA){
			#if (VERBOSE)
				fprintf(stderr,"Host[%d]=%f, Device[%d]=%f, Difference=%f, max allowed difference=%f\n" \
							  ,i \
							  ,host[i] \
							  ,i \
							  ,device[i] \
							  ,(host[i]-device[i]) \
							  ,DELTA \
							  );
			#endif
			return 1;		
		}
	}			
	return 0;
}



