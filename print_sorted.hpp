#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>

#ifndef PRINT_SORTED_GUARD
#define PRINT_SORTED_GUARD


void print_sorted(std::vector<int> &letter_occurrences){
   

    std::vector<std::pair<u_char,int>> letter_occurrences_sorted(256);
    for (int i=0;i<256;i++){
        letter_occurrences_sorted[i]={(u_char)i,letter_occurrences[i]};
    }
    std::sort(letter_occurrences_sorted.begin(),letter_occurrences_sorted.end(),
                [](auto x1,auto x2){return x1.second > x2.second;});


    int total_num_characters=std::reduce(letter_occurrences.begin(),letter_occurrences.end());
    for (auto [character, occurrences] : letter_occurrences_sorted) {
        if (occurrences!=0){
            float percentage=100*(float)occurrences/total_num_characters;
            std::printf("%c (%.2f %)\n", character, percentage);
        }
    }
}



#endif