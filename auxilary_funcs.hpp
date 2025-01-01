#include <algorithm>
#include <fstream>
#include <numeric>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <unordered_map>

#ifndef PRINT_SORTED_GUARD
#define PRINT_SORTED_GUARD

std::string open_file_as_string(std::string filename){
    auto file = std::ifstream(filename);
    if (!file.is_open()) {
        throw std::invalid_argument("File doesn't exist or it couldn't be opened");
    }

    std::vector<int> letter_occurrences(256, 0);
    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return file_content;
}

void save_words_sorted_to_file(const std::unordered_map<std::string,int> &histogram,
                              const std::string filename){
    std::vector<std::pair<std::string, int>> vec_sorted (histogram.begin(), histogram.end());

    std::sort(vec_sorted.begin(), vec_sorted.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: sorted_histogram.txt\n";
        return;
    };

    double num_words =  std::accumulate(vec_sorted.begin(), vec_sorted.end(), 0,
    [](int sum, const std::pair<std::string, int>& p) {return sum + p.second;});


    for (auto [word,occurrences] : vec_sorted){
        float percentage=100.0f*occurrences/num_words;
        outfile<<word<<" = "<<percentage<<"% ("<<occurrences<<")\n";
    }


}



void save_letters_sorted_to_file(const std::vector<int> &histogram, std::string filename) {
    std::vector<std::pair<u_char, int>> histogram_sorted(256);
    for (int i = 0; i < 256; i++) {
        histogram_sorted[i] = {(u_char)i, histogram[i]};
    }
    std::sort(histogram_sorted.begin(), histogram_sorted.end(),
              [](auto x1, auto x2) { return x1.second > x2.second; });

    int total_num_characters = std::reduce(histogram.begin(), histogram.end());
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: sorted_histogram.txt\n";
        return;
    }
    for (auto [character, occurrences] : histogram_sorted) {
        if (occurrences != 0) {
            float percentage = 100 * (float)occurrences / total_num_characters;
            outfile << character << " (" << percentage << " %)" << std::endl;
        }
    };
}



class Timer {
public:
    Timer(const std::string& message) : message(message), start(std::chrono::high_resolution_clock::now()) {}

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << message <<  duration.count() << " ms" << std::endl;
    }

    std::string message;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::duration<double, std::milli> duration;
};


#endif