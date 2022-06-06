/**
 * @file testMatrixConv.cu
 * @brief test program to profile convolution layer kernels
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "./inc/common.cuh"
#include "./inc/convolutionLayerFunc.cuh"
#include <string.h>

int main(int argc, char **argv){
	
	int iKernel;
	int blockx=16;
	int blocky=16;
	int i,j,tile_size;
	
	deviceInfor();
	
	if (argc<2){
		fprintf(stderr,"usage: <%s> <iKernel> [optional <blockx>] [optional <blocky>]\n",argv[0]);
	    fprintf(stderr,"ikernel=0: naiveConvLayer\n");
	    fprintf(stderr,"ikernel=1: sharedConvLayer\n");
	    fprintf(stderr,"ikernel=2: sharedUnroll5ConvLayer\n");
	    fprintf(stderr,"ikernel=3: approx5ConvLayer\n");
	    fprintf(stderr,"ikernel=4: custom5ConvLayer\n");
		exit(1);
	}
	
	iKernel=atoi(argv[1]);
	if (argc>2) blockx=atoi(argv[2]);
	if (argc>3) blocky=atoi(argv[3]);
	
	
	dim3 block (blockx, blocky);
    dim3 grid  ((NXY + block.x - 1) / block.x, (NXY + block.y - 1) / block.y);
    double iStart, iElaps;
    double effBW;
    double iGain = 0.0;
    
    int odata_w=NXY-FILTER5+1;
    int odata_h=NXY-FILTER5+1;
    
    //data
    float *hSource, *hDest;
    float *dSource, *dDest;
    float *gpuRes;
    float *hFilter;
    float *dFilter;
    //alloc on host
    hSource=(float *)malloc(NXY*NXY*sizeof(float));
    CHECK_PTR(hSource);
    hDest=(float *)malloc(odata_w*odata_h*sizeof(float));
    CHECK_PTR(hDest);
    hFilter=(float *)malloc(FILTER5*FILTER5*sizeof(float));
    CHECK_PTR(hFilter);
    gpuRes=(float *)malloc(odata_h*odata_w*sizeof(float));
    CHECK_PTR(gpuRes);
    //alloc on device
    CHECK_CUDA(cudaMalloc( (void**)&dSource, NXY*NXY*sizeof(float)));	
    CHECK_CUDA(cudaMalloc( (void**)&dDest, odata_h*odata_w*sizeof(float)));
    CHECK_CUDA(cudaMalloc( (void**)&dFilter, FILTER5*FILTER5*sizeof(float)));
    //init on host
    for(i=0;i<NXY*NXY;i++)
    	hSource[i]=randomUint8()/(float)1.145f;
    for(i=0;i<FILTER5*FILTER5;i++)
    	hFilter[i]=0.0;//randomUint8()/(float)13.561f;	   	
    hFilter[12]=1;		
    
    #if (VERBOSE)	
    	fprintf(stderr,"Source:\n");
        for(i=0;i<NXY;i++){
        	for(j=0;j<NXY;j++){
        		fprintf(stderr,"%.1f ",hSource[i*NXY + j]);	
        	}
        	fprintf(stderr,"\n");
        }
        fprintf(stderr,"Kernel:\n");
        for(i=0;i<FILTER5;i++){
        	for(j=0;j<FILTER5;j++){
        		fprintf(stderr,"%.1f ",hFilter[i*FILTER5 + j]);	
        	}
        	fprintf(stderr,"\n");
        }
    #endif
    	
    //copy on GPU
    CHECK_CUDA(cudaMemcpy(dSource, hSource, NXY*NXY*sizeof(float), cudaMemcpyHostToDevice));	
    CHECK_CUDA(cudaMemcpy(dFilter, hFilter, FILTER5*FILTER5*sizeof(float), cudaMemcpyHostToDevice));	
    char *kernelName;
    
	    
    switch(iKernel){
    	/*setup naiveConvLayer*/
    	case 0:
    		#if (VERBOSE)
    			fprintf(stdout,"naiveConvLayer kernel selected\n");
    		#endif
    		kernelName=strdup("naiveConvLayer");
    		iStart = cpuSecond();
    		naiveConvLayer<<<grid,block>>>(dSource, dDest, dFilter, FILTER5);
    		CHECK_CUDA(cudaGetLastError());
			CHECK_CUDA(cudaDeviceSynchronize());
   			iElaps = cpuSecond() - iStart;
    		break;
    	/*setup sharedConvLayer*/
    	case 1:
    		#if (VERBOSE)
    			fprintf(stdout,"sharedConvLayer kernel selected\n");
    		#endif
    		kernelName=strdup("sharedConvLayer");
   			tile_size = blockx;
    		block.x = (blockx + FILTER5 -1);
    		block.y = (blockx + FILTER5 -1);
    		grid.x = (NXY + tile_size - 1) / tile_size;
    		grid.y = (NXY + tile_size - 1) / tile_size;
    		iStart = cpuSecond();
    		sharedConvLayer<<<grid,block,(block.x*block.y*sizeof(float))>>>(dSource, dDest, dFilter, FILTER5);
    		CHECK_CUDA(cudaGetLastError());
			CHECK_CUDA(cudaDeviceSynchronize());
   			iElaps = cpuSecond() - iStart;
    		break;
    	/*setup sharedUnroll5ConvLayer*/
    	case 2:
    		#if (VERBOSE)
    			fprintf(stdout,"sharedUnroll5ConvLayer kernel selected\n");
    		#endif
    		kernelName=strdup("sharedUnroll5ConvLayer");
   			tile_size = blockx;
    		block.x = (blockx + FILTER5 -1);
    		block.y = (blockx + FILTER5 -1);
    		grid.x = (NXY + tile_size - 1) / tile_size;
    		grid.y = (NXY + tile_size - 1) / tile_size;
    		iStart = cpuSecond();
    		sharedUnroll5ConvLayer<<<grid,block,(block.x*block.y*sizeof(float))>>>(dSource, dDest, dFilter);
    		CHECK_CUDA(cudaGetLastError());
			CHECK_CUDA(cudaDeviceSynchronize());
   			iElaps = cpuSecond() - iStart;
   			break;
    	/*setup approx5ConvLayer*/
    	case 3:
    		#if (VERBOSE)
    			fprintf(stdout,"approx5ConvLayer kernel selected\n");
    		#endif
    		kernelName=strdup("approx5ConvLayer");
   			tile_size = blockx;
    		block.x = (blockx + FILTER5 -1);
    		block.y = (blockx + FILTER5 -1);
    		grid.x = (NXY + tile_size - 1) / tile_size;
    		grid.y = (NXY + tile_size - 1) / tile_size;
    		iStart = cpuSecond();
    		approx5ConvLayer<<<grid,block,(block.x*block.y*sizeof(float))>>>(dSource, dDest, dFilter);
    		CHECK_CUDA(cudaGetLastError());
			CHECK_CUDA(cudaDeviceSynchronize());
   			iElaps = cpuSecond() - iStart;
    		break;
    	/*setup custom5ConvLayer*/
    	case 4:
    		#if (VERBOSE)
    			fprintf(stdout,"custom5ConvLayer kernel selected\n");
    		#endif
    		kernelName=strdup("custom5ConvLayer");
    		block.x = NXY;
    		block.y = NXY;
    		grid.x = 1;
    		grid.y = 1;
    		iStart = cpuSecond();
    		custom5ConvLayer<<<grid,block,block.x*block.y*sizeof(float)>>>(dSource, dDest, dFilter);
    		CHECK_CUDA(cudaGetLastError());
			CHECK_CUDA(cudaDeviceSynchronize());
   			iElaps = cpuSecond() - iStart;
    		break;
    	default:
    		#if (VERBOSE)
    		fprintf(stderr,"error in kernel selection\n");
    		#endif
    		exit(1);
    		break;
    }
    
    #if (VERBOSE)
    	fprintf(stdout,"nx=%d, ny=%d, %lu Bytes, grid(%d,%d), block(%d,%d), #threads=%llu\n",NXY,NXY,
    			(NXY*NXY*sizeof(float)),grid.x,grid.y,
    			 block.x,block.y,(long long unsigned int)(block.x*block.y*grid.x*grid.y));
    #endif

    
    //get data back from gpu
    CHECK_CUDA(cudaMemcpy(gpuRes, dDest, odata_h*odata_w*sizeof(float), cudaMemcpyDeviceToHost));
    
    #if (CHECK)
    	//compute result on host
    	iStart = cpuSecond();
    	hostConvLayer(hSource, hDest, hFilter, FILTER5);
    	iGain = cpuSecond() - iStart;
    	iGain = iGain/iElaps;
    	// check kernel results
    	#if (VERBOSE)
    		fprintf(stderr,"CPU result:\n");
        	for(i=0;i<odata_h;i++){
        		for(j=0;j<odata_w;j++){
        			fprintf(stderr,"%.2f ",hDest[i*odata_w + j]);	
        		}
        		fprintf(stderr,"\n");
        	}
        	fprintf(stderr,"GPU result:\n");
        	for(i=0;i<odata_h;i++){
        		for(j=0;j<odata_w;j++){
        			fprintf(stderr,"%.2f ",gpuRes[i*odata_w + j]);	
        		}
        		fprintf(stderr,"\n");
        	}
    	#endif
        if(checkRes(hDest,gpuRes,odata_w,odata_h)==1){
        	fprintf(stderr,"GPU and CPU result missmatch!\n");		
        	exit(1);
        }
    #endif

    // calculate effective_bandwidth (MB/s)
    effBW=(2 * NXY * NXY * sizeof(float)) / ((1e+6f)*iElaps);
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



