#include <fstream>
#include <sstream>
#include <iostream>
#include <exception>
#include <string>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include "auxilary_funcs.hpp"


class increment
{   
    public:
        int* d_histogram_ptr;
        increment(thrust::device_vector<int> &d_histogram){
            d_histogram_ptr=thrust::raw_pointer_cast(d_histogram.data());
        }

        __device__
        void operator()(u_char character){
            //d_letter_occurrences_ptr[character]++; //would cause race conditions
            atomicAdd(&d_histogram_ptr[character],1);
        }
};

int main(){  
    Timer time_all("Time execution (GPU-Thrust)");
    std::string h_text=open_file_as_string("sherlock_holmes_canon.txt");
    //moving to the device
    thrust::device_vector<u_char> d_text(h_text.begin(), h_text.end());
    thrust::device_vector<int> d_histogram(256, 0);
    thrust::for_each(d_text.begin(), d_text.end(), increment(d_histogram));
    
    //moving to the host
    std::vector<int> h_histogram(d_histogram.size());
    thrust::copy(d_histogram.begin(), d_histogram.end(), h_histogram.begin());
    //thrust::host_device<int> h_letter_occurrences=d_letter_occurrences also works
    time_all.stop();
    save_sorted_to_file(h_histogram);
}
