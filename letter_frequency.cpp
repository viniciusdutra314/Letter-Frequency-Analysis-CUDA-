#include <fstream>
#include <iostream>
#include <exception>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include "print_sorted.hpp"



int main(){

    auto file=std::ifstream("sherlock_homes_canon.txt");
    if (!file.is_open()){
        throw std::invalid_argument("File doesn't exist or it couldn't been opened");
    }

    std::vector<int> letter_occurrences(256);
    std::string line_text;
    while (getline(file,line_text)){
        for (u_char character:line_text){
            if (!isspace(character)){
                letter_occurrences[character]++;
            }   
        }
    }
    print_sorted(letter_occurrences);
}