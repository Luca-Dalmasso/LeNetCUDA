/**
 * @file testMatrixConv.cu
 * @brief test program to profile matrixConvolution kernels
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "./inc/common.cuh"
#include "./inc/matrix2Dconvolution.cuh"
#include <string.h>


int main(int argc, char **argv){
	#if (VERBOSE)
		deviceInfor();
	#endif
	
	int iKernel;
	uint_32 nx=1<<10;
	uint_32 ny=1<<10;
	uint_32 blockx=16;
	uint_32 blocky=16;
	int i,j,tile_size;
	
	if (argc<2){
		fprintf(stderr,"usage: <%s> <iKernel> [optional <blockx>] [optional <blocky>] [optional <nx>] [optional <ny>]\n",argv[0]);
	    fprintf(stderr,"ikernel=0: naive2Dconvolution\n");
	    fprintf(stderr,"ikernel=1: naiveUnroll2Dconvolution\n");
	    fprintf(stderr,"ikernel=2: shared2Dconvolution\n");
	    fprintf(stderr,"ikernel=3: sharedUnroll2Dconvolution\n");
	    fprintf(stderr,"ikernel=4: fm_sharedUnroll2Dconvolution\n");
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
    		
    
    #if (VERBOSE)	
    	if(nx<=32 && ny<=32){
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
        }
    #endif
    	
    //copy on GPU
    CHECK_CUDA(cudaMemcpy(dSource, hSource, nx*ny*sizeof(float), cudaMemcpyHostToDevice));	
    CHECK_CUDA(cudaMemcpy(dFilter, hFilter, KERNEL_SIZE*KERNEL_SIZE*sizeof(float), cudaMemcpyHostToDevice));	
    
    
    void (* kernel) (float *, float *, int, int, float *, int);
    char *kernelName;
    
    switch(iKernel){
    	/*setup naive2Dconvolution*/
    	case 0:
    		#if (VERBOSE)
    			fprintf(stdout,"naive2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("naive2Dconvolution");
    		kernel=&naive2Dconvolution;
    		break;
    	/*setup naiveUnroll2Dconvolution*/
    	case 1:
    		#if (VERBOSE)
    			fprintf(stdout,"naiveUnroll2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("naiveUnroll2Dconvolution");
    		kernel=&naiveUnroll2Dconvolution;
    		break;
    	/*setup shared2Dconvolution*/
    	case 2:
    		#if (VERBOSE)
    			fprintf(stdout,"shared2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("shared2Dconvolution");
    		kernel=&shared2Dconvolution;
   			tile_size = blockx;
    		block.x = (blockx + KERNEL_SIZE -1);
    		block.y = (blockx + KERNEL_SIZE -1);
    		grid.x = (nx + tile_size - 1) / tile_size;
    		grid.y = (ny + tile_size - 1) / tile_size;
    		break;
    	/*setup sharedUnroll2Dconvolution*/
    	case 3:
    		#if (VERBOSE)
    			fprintf(stdout,"sharedUnroll2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("sharedUnroll2Dconvolution");
    		kernel=&sharedUnroll2Dconvolution;
   			tile_size = blockx;
    		block.x = (blockx + KERNEL_SIZE -1);
    		block.y = (blockx + KERNEL_SIZE -1);
    		grid.x = (nx + tile_size - 1) / tile_size;
    		grid.y = (ny + tile_size - 1) / tile_size;
    		break;
    	/*setup fm_sharedUnroll2Dconvolution*/
    	case 4:
    		#if (VERBOSE)
    			fprintf(stdout,"fm_sharedUnroll2Dconvolution kernel selected\n");
    		#endif
    		kernelName=strdup("fm_sharedUnroll2Dconvolution");
    		kernel=&fm_sharedUnroll2Dconvolution;
   			tile_size = blockx;
    		block.x = (blockx + KERNEL_SIZE -1);
    		block.y = (blockx + KERNEL_SIZE -1);
    		grid.x = (nx + tile_size - 1) / tile_size;
    		grid.y = (ny + tile_size - 1) / tile_size;
    		break;
    	default:
    		#if (VERBOSE)
    		fprintf(stderr,"error in kernel selection\n");
    		#endif
    		exit(1);
    		break;
    }
    
    #if (VERBOSE)
    	fprintf(stdout,"nx=%d, ny=%d, %lu Bytes, grid(%d,%d), block(%d,%d), #threads=%llu\n",nx,ny,
    			(nx*ny*sizeof(float)),grid.x,grid.y,
    			 block.x,block.y,(long long unsigned int)(block.x*block.y*grid.x*grid.y));
    #endif
    
    if (iKernel==2 || iKernel ==3 || iKernel==4 || iKernel==5){
    	//dynamic shared memory kernels
    	iStart = cpuSecond();
    	kernel<<<grid,block,(block.x*block.y*sizeof(float))>>>(dSource, dDest, nx, ny, dFilter, KERNEL_SIZE);
    	CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
   		iElaps = cpuSecond() - iStart;
   	}else{
   		//standard kernels with no shared memory
   		iStart = cpuSecond();
    	kernel<<<grid,block>>>(dSource, dDest, nx, ny, dFilter, KERNEL_SIZE);
    	CHECK_CUDA(cudaGetLastError());
		CHECK_CUDA(cudaDeviceSynchronize());
   		iElaps = cpuSecond() - iStart;
   	}
    
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
        	//exit(1);
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



