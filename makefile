
#Author: Dalmasso Luca
#Last modification: 09/06/2022 by Luca

#Maxwell cards
#SM_50: Tesla/Quadro M series
#SM_52: Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X.
#SM_53: Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano.
GPU_ARCHITECTURE=sm_53
#Application name (main.cu file by default)
NAME=LeNet
SRC=src/
INCLUDES=inc/
BUILD=build/
DOCS=docs/
NVCC=/usr/local/cuda/bin/nvcc
CUDA_FLAGS=--resource-usage
CUDA_FLAGS+=-arch=sm_53
DEBUG=0
FMATH=0

ifeq ($(DEBUG),1)
	CUDA_FLAGS+=-g
	CUDA_FLAGS+=-G
	CUDA_FLAGS+=-DDEBUG
else
	CUDA_FLAGS+=-O2	
endif

ifeq ($(FMATH),1)
	CUDA_FLAGS+=--use_fast_math
endif

.PHONY: all clean docs test

SOURCES=$(wildcard $(SRC)*.cu)
BUILD_PATH=$(subst $(SRC),$(BUILD),$(SOURCES))
OBJECTS=$(patsubst %.cu, %.o, $(BUILD_PATH))
HEADERS=$(wildcard $(INCLUDES)*.cuh)
	
all: $(BUILD)$(NAME).o $(OBJECTS)
	@echo $@;
	$(NVCC)  $^ -o $(NAME)
	@echo "Generating ptx for main application..";
	$(NVCC) $(CUDA_FLAGS) --ptx -I$(INCLUDES) $(NAME).cu

$(BUILD)$(NAME).o: $(NAME).cu
	@echo $@;
	$(NVCC) $(CUDA_FLAGS) -I$(INCLUDES) -c $< -o $@

$(BUILD)%.o: $(SRC)%.cu $(HEADERS)
	@echo $@;
	$(NVCC) $(CUDA_FLAGS) -I$(INCLUDES) -c $< -o $@

#docs: ./Doxyfile $(HEADERS) $(SOURCES) $(NAME).cu
#	@echo $@;
#	doxygen ./Doxyfile

test:
	@echo "SOURCES: $(SOURCES)";
	@echo "OBJECTS: $(OBJECTS)";
	@echo "HEADERS: $(HEADERS)";
	@echo "CUDA FLAGS: $(CUDA_FLAGS)";

clean:
	@echo $@;
	rm -rf $(BUILD)*
	rm -f $(NAME).o
	rm -f $(NAME)
	rm -f $(NAME).ptx

	
