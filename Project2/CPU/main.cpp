using namespace std;

#include <iostream>
#include <fstream>
#include <string>
#include <regex>


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Need two arguments " << argv[0] << " <text> <regex list>" << std::endl;
        return 1;
    }

    std::cout << "Argument 1: " << argv[1] << std::endl;
    std::cout << "Argument 2: " << argv[2] << std::endl;


    
    ifstream text_file(argv[1]);
    if (!text_file) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return 1;
    }

    ifstream regexes_file(argv[2]);
    if (!regexes_file) {
        std::cerr << "Error opening file: " << argv[2] << std::endl;
        return 1;
    }

    std::string text_content;
    std::string text_line;
    while (getline(text_file, text_line)) {
        text_content += text_line + "\n";
    }
    text_file.close();

    std::string line;
    while (getline(regexes_file, line)) {
        if (std::regex_search (text_content, std::regex(line))){
            std::cout << "regex matched: " << line << std::endl;
        } else {
            std::cout << "regex not matched: " << line << std::endl;
        }
    }

    regexes_file.close(); 
    return 0;
}