#include <string>
#include <vector>
#include "auxilary_funcs.hpp"



int main(){
    std::string file_str=open_file_as_string("sherlock_holmes_canon.txt");
    std::vector<int> histogram(256);

    Timer time_all("Time of execution (CPU)");
    for (u_char character : file_str) {
        histogram[character]++;
    }

    time_all.stop();
    save_sorted_to_file(histogram);
}