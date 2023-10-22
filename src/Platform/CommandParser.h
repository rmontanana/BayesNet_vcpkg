#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H
#include <string>
#include <vector>
#include <tuple>
using namespace std;

namespace platform {
    class CommandParser {
    public:
        CommandParser() = default;
        pair<char, int> parse(const string& color, const vector<tuple<string, char, bool>>& options, const char defaultCommand);
        char getCommand() const { return command; };
        int getIndex() const { return index; };
    private:
        void messageError(const string& message);
        char command;
        int index;
    };
} /* namespace platform */
#endif /* COMMAND_PARSER_H */