#include <string>
#include <vector>
#include "../auxilary_funcs.hpp"
#include <unordered_map>
#include <string>
#include <iostream>


int main(){
    Timer time_all("(CPU) Total time: ");
    std::string file_str=open_file_as_string("../sherlock_holmes_canon.txt");
    time_all.stop();
    std::unordered_map<std::string,int> histogram;
    std::string partial_word;
    for (u_char character:file_str){
        if (isalpha(character)){
            partial_word+=tolower(character);
        }   
        else if (isspace(character) and !partial_word.empty()){
            histogram[partial_word]++;
            partial_word.clear();
        }
    };

    save_words_sorted_to_file(histogram,"word_histogram_cpu.txt");

    time_all.stop();
}