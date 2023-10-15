#ifndef UTILS_H
#define UTILS_H
#include <sstream>
#include <string>
#include <vector>
namespace platform {
    //static vector<string> split(const string& text, char delimiter);
    static std::vector<std::string> split(const std::string& text, char delimiter)
    {
        std::vector<std::string> result;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            result.push_back(token);
        }
        return result;
    }
}
#endif