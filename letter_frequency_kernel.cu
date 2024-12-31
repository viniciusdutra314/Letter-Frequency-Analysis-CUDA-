#include <string>
#include "auxilary_funcs.hpp"

__global__ void kernel_histogram(u_char* text, int text_size,int* histogram)
{

    __shared__ int local_occurrences[256];
    local_occurrences[threadIdx.x]=0;
    __syncthreads();

    int position_in_the_text = threadIdx.x + blockIdx.x * blockDim.x;
    int step_to_next_position= blockDim.x * gridDim.x;
    while (position_in_the_text < text_size)
    {
        atomicAdd(&local_occurrences[text[position_in_the_text]], 1);
        position_in_the_text += step_to_next_position;
    }
    __syncthreads();
    atomicAdd(&(histogram[threadIdx.x]), local_occurrences[threadIdx.x]);
}

int main(){  
    std::string h_text=open_file_as_string("sherlock_holmes_canon.txt");
    //moving to the device
    Timer time_all("Entire execution");
    u_char* d_text;
    int* d_histogram;
    int memory_used_text=h_text.size()*sizeof(u_char);
    int memory_used_hist=256*sizeof(int);
    cudaMalloc(&d_text,memory_used_text);
    cudaMemcpy(d_text,h_text.c_str(),memory_used_text,cudaMemcpyHostToDevice);

    cudaMalloc(&d_histogram,memory_used_hist);
    cudaMemset(d_histogram,0,memory_used_hist);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;
    Timer time_kernel("Kernel");
    kernel_histogram<<<blocks * 2, 256>>>(d_text,h_text.size(),d_histogram);
    time_kernel.stop();
    int* h_histogram=(int*) malloc(memory_used_hist);
    cudaMemcpy(h_histogram,d_histogram,memory_used_hist,cudaMemcpyDeviceToHost);


    time_all.stop();
    
    std::vector<int> histogram_vector(h_histogram,h_histogram+256);
    save_sorted_to_file(histogram_vector);
    free(h_histogram);
    cudaFree(d_text); 
    cudaFree(d_histogram);
 
    
}
