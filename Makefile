MAKEFLAGS += --silent

compile:
	mkdir -p bin
	echo "Compiling..."
	g++ -O3 letter_frequency.cpp -o bin/cpu
	nvcc -O3 letter_frequency_thrust.cu -o bin/gpu_thrust
	nvcc -O3 letter_frequency_kernel.cu -o bin/gpu_kernel
compile_and_run:
	make compile
	bin/cpu
	bin/gpu_thrust
	bin/gpu_kernel
run:
	bin/cpu
	bin/gpu_thrust
	bin/gpu_kernel
	#checking the results
	diff3 output/histogram_cpu.txt output/histogram_gpu_thrust.txt output/histogram_gpu_kernel.txt