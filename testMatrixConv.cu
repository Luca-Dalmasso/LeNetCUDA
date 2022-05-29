/**
 * @file testMatrixConv.cu
 * @brief test program to profile matrixConvolution kernels
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "./inc/common.h"
#include "./inc/matrix2Dconvolution.h"
#include <string.h>

#define KERNEL_SIZE 5

int main(int argc, char **argv){
	#if (VERBOSE)
		deviceInfor();
	#endif
	
	int iKernel;
	uint_32 nx=1<<12;
	uint_32 ny=1<<12;
	uint_32 blockx=16;
	uint_32 blocky=16;
	uint_32 i,j;
	
	if (argc<2){
		fprintf(stderr,"usage: <%s> <iKernel> [optional <blockx>] [optional <blocky>] [optional <nx>] [optional <ny>]\n",argv[0]);
	    fprintf(stderr,"ikernel=0: naive2Dconvolution\n");
		exit(1);
	}
	
	iKernel=atoi(argv[1]);
	if (argc>2) blockx=atoi(argv[2]);
	if (argc>3) blocky=atoi(argv[3]);
	if (argc>4) nx=atoi(argv[4]);
	if (argc>5) ny=atoi(argv[5]);
	
	dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    double iStart, iElaps;
    double effBW;
    double iGain = 0.0;
    
    //data
    float *hSource, *hDest;
    float *dSource, *dDest;
    float *gpuRes;
    float *hFilter;
    float *dFilter;
    //alloc on host
    hSource=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(hSource);
    hDest=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(hDest);
    hFilter=(float *)malloc(KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
    CHECK_PTR(hFilter);
    gpuRes=(float *)malloc(nx*ny*sizeof(float));
    CHECK_PTR(gpuRes);
    //alloc on device
    CHECK_CUDA(cudaMalloc( (void**)&dSource, nx*ny*sizeof(float)));	
    CHECK_CUDA(cudaMalloc( (void**)&dDest, nx*ny*sizeof(float)));
    CHECK_CUDA(cudaMalloc( (void**)&dFilter, KERNEL_SIZE*KERNEL_SIZE*sizeof(float)));
    //init on host
    for(i=0;i<nx*ny;i++)
    	hSource[i]=randomUint8()/(float)1.145f;
    for(i=0;i<KERNEL_SIZE*KERNEL_SIZE;i++)
    	hFilter[i]=randomUint8()/(float)13.561f;
    //copy on GPU
    CHECK_CUDA(cudaMemcpy(dSource, hSource, nx*ny*sizeof(float), cudaMemcpyHostToDevice));	
    CHECK_CUDA(cudaMemcpy(dFilter, hFilter, KERNEL_SIZE*KERNEL_SIZE*sizeof(float), cudaMemcpyHostToDevice));	
    
    #if (VERBOSE)
    fprintf(stdout,"nx=%d, ny=%d, %lu Bytes, grid(%d,%d), block(%d,%d), #threads=%llu\n",nx,ny,
    			(nx*ny*sizeof(float)),grid.x,grid.y,
    			 block.x,block.y,(long long unsigned int)(block.x*block.y*grid.x*grid.y));
    #endif
    
    void (* kernel) (float *, float *, uint_32, uint_32, float *, uint_32);
    char *kernelName;
    
    switch(iKernel){
    	/*setup */
    	case 0:
    		#if (VERBOSE)
    		fprintf(stdout,"naive2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("naive2Dconvolution ");
    		kernel=&naive2Dconvolution;
    		break;
    	
    	default:
    		#if (VERBOSE)
    		fprintf(stderr,"error in kernel selection\n");
    		#endif
    		exit(1);
    		break;
    }
    iStart = cpuSecond();
    kernel<<<grid,block>>>(dSource, dDest, nx, ny, dFilter, KERNEL_SIZE);
    CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaDeviceSynchronize());
   	iElaps = cpuSecond() - iStart;
    
    //get data back from gpu
    CHECK_CUDA(cudaMemcpy(gpuRes, dDest, nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
    
    #if (CHECK)
    	//compute result on host
    	iStart = cpuSecond();
    	host2Dconvolution(hSource, hDest, nx, ny, hFilter, KERNEL_SIZE);
    	iGain = cpuSecond() - iStart;
    	iGain = iGain/iElaps;
    	// check kernel results
    	#if (VERBOSE)
    		if (nx<=32 && ny<=32){
    			fprintf(stderr,"Source:\n");
        		for(i=0;i<nx;i++){
        			for(j=0;j<ny;j++){
        				fprintf(stderr,"%.1f ",hSource[i*nx + j]);	
        			}
        			fprintf(stderr,"\n");
        		}
        		fprintf(stderr,"Kernel:\n");
        		for(i=0;i<KERNEL_SIZE;i++){
        			for(j=0;j<KERNEL_SIZE;j++){
        				fprintf(stderr,"%.1f ",hFilter[i*KERNEL_SIZE + j]);	
        			}
        			fprintf(stderr,"\n");
        		}
    			fprintf(stderr,"CPU result:\n");
        		for(i=0;i<nx;i++){
        			for(j=0;j<ny;j++){
        				fprintf(stderr,"%.1f ",hDest[i*nx + j]);	
        			}
        			fprintf(stderr,"\n");
        		}
        		fprintf(stderr,"GPU result:\n");
        		for(i=0;i<nx;i++){
        			for(j=0;j<ny;j++){
        				fprintf(stderr,"%.1f ",gpuRes[i*nx + j]);	
        			}
        			fprintf(stderr,"\n");
        		}
    		}
    	#endif
        if(checkRes(hDest,gpuRes,nx,ny)==1){
        	fprintf(stderr,"GPU and CPU result missmatch!\n");		
        	exit(1);
        }
    #endif

    // calculate effective_bandwidth (MB/s)
    effBW=(2 * nx * ny * sizeof(float)) / ((1e+6f)*iElaps);
    /*printf on stdout used for profiling <kernelName>,<elapsedTime>,<bandwidth>,<gain>,<grid(x,y)>,<block(x,y)>*/
    fprintf(stdout,"%s,%f,%f,%f,grid(%d.%d),block(%d.%d)\n",kernelName, effBW, iElaps, iGain, grid.x, grid.y, block.x, block.y);

    // free host and device memory
    CHECK_CUDA(cudaFree(dSource));
    CHECK_CUDA(cudaFree(dDest));
	CHECK_CUDA(cudaFree(dFilter));
	free(hSource);
    free(hDest);
    free(hFilter);
    free(gpuRes);
    
    // reset device
    CHECK_CUDA(cudaDeviceReset());
	
	return 0;
}



