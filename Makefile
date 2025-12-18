# Compiler and flags
NVCC := nvcc
FLAGS := -O3 -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lnvToolsExt --diag-suppress 177

# Target executable name
TARGET := build/gemm_binary

# Find all .cu files in the current directory
SOURCES := $(wildcard src/*.cu)

# Default rule
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(FLAGS) $(SOURCES) -o $(TARGET)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)