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
#include "print_sorted.hpp"


class increment
{   
    public:
        int* d_letter_occurrences_ptr;
        increment(thrust::device_vector<int> &d_letter_occurrences){
            d_letter_occurrences_ptr=thrust::raw_pointer_cast(d_letter_occurrences.data());
        }

        __device__
        void operator()(u_char character){
            //d_letter_occurrences_ptr[character]++; //would cause race conditions
            atomicAdd(&d_letter_occurrences_ptr[character],1);
        }
};

int main(){  
    auto file = std::ifstream("sherlock_homes_canon.txt");
    if (!file.is_open()){
        throw std::invalid_argument("File doesn't exist or it couldn't been opened");
    }
    //copying entire text file
    std::ostringstream oss;
    oss << file.rdbuf();
    std::string h_text = oss.str();
    //moving to the device
    thrust::device_vector<u_char> d_text(h_text.begin(), h_text.end());
    thrust::device_vector<int> d_letter_occurrences(256, 0);
    thrust::for_each(d_text.begin(), d_text.end(), increment(d_letter_occurrences));
    
    //moving to the host
    std::vector<int> letter_occurrences(d_letter_occurrences.size());
    thrust::copy(d_letter_occurrences.begin(), d_letter_occurrences.end(), letter_occurrences.begin());
    //thrust::host_device<int> h_letter_occurrences=d_letter_occurrences also works
    print_sorted(letter_occurrences);
}
