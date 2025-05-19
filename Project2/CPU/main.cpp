#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Need two arguments " << argv[0] << " <text> <regex list>" << std::endl;
        return 1;
    }

    std::cout << "Argument 1: " << argv[1] << std::endl;
    std::cout << "Argument 2: " << argv[2] << std::endl;

    return 0;
}