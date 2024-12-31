#include <string>
#include <vector>
#include "auxilary_funcs.hpp"



int main(){
    Timer time_all("(CPU) Total time: ");
    std::string file_str=open_file_as_string("input.txt");
    std::vector<int> histogram(256);

    Timer time_processing("(CPU) Processing: ");
    for (u_char character : file_str) {
        histogram[character]++;
    }

    time_processing.stop();
    time_all.stop();
   
    std::cout<<std::endl;

    save_sorted_to_file(histogram,"output/histogram_cpu.txt");
}