#include <fstream>
#include <iostream>
#include <exception>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std; //just to make easier to read


int main(){

    vector<int> letter_occurrences(256);


    auto file=ifstream("sherlock_homes_canon.txt");

    if (!file.is_open()){
        throw invalid_argument("File doesn't exist or it couldn't been opened");
    }

    string line_text;
    while (getline(file,line_text)){
        for (u_char character:line_text){
            if (!isspace(character)){
                letter_occurrences[character]++;
            }   
        }
    }
    int total_num_characters=reduce(letter_occurrences.begin(),letter_occurrences.end());

    vector<pair<u_char,int>> letter_occurrences_sorted(256);
    for (int i=0;i<256;i++){
        letter_occurrences_sorted[i]={(u_char)i,letter_occurrences[i]};
    }
    sort(letter_occurrences_sorted.begin(),letter_occurrences_sorted.end(),
                [](auto x1,auto x2){return x1.second > x2.second;});

    for (auto [character, occurrences] : letter_occurrences_sorted) {
        if (occurrences!=0){
            float percentage=100*(float)occurrences/total_num_characters;
            printf("%c (%.2f %)\n", character, percentage);
        }
    }
}