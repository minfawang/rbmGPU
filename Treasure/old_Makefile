CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# CUDA code generation flags
# GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := -arch=sm_35

# extra NVCC flags
# EXTRA_NVCCFLAGS := --relocatable-device-code true

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcurand # -lcudadevrt
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcurand # -lcudadevrt
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcurand # -lcudadevrt
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -lcublas -lcurand -lcudadevrt --relocatable-device-code true # -m32
else
      NVCCFLAGS := -lcublas -lcurand -lcudadevrt --relocatable-device-code true # -m64
endif

TARGETS = rbm

all: $(TARGETS)

rbm: rbm.cc rbm.o
	$(CC) $< -std=c++0x -o $@ rbm.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

rbm.o: rbm_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
