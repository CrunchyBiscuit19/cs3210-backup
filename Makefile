NVCC ?= nvcc
NVCCFLAGS ?= -g -std=c++17 -lineinfo -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=sm_80 -Xcompiler -O3

HEADERS := geometry.cuh graphics.cuh utils.cuh

# Every versions/v*.cu is a self-contained render_scene implementation (it can be pasted
# into path_tracing.cu as is), built as build/<name>. -I. lets it find the provided headers.
VERSIONS := $(basename $(notdir $(wildcard versions/v*.cu)))

.DEFAULT_GOAL := path_tracing
.PHONY: all clean versions
all: cpu_path_tracing path_tracing versions

cpu_path_tracing: main.cu cpu_path_tracing.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp main.cu cpu_path_tracing.cu -o $@

path_tracing: main.cu path_tracing.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) main.cu path_tracing.cu -o $@

versions: $(addprefix build/,$(VERSIONS))

# -Xptxas -v prints registers, spills and shared/constant memory per kernel.
build/%: main.cu versions/%.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -I. -Xptxas -v main.cu versions/$*.cu -o $@

clean:
	rm -f cpu_path_tracing path_tracing
	rm -rf build
