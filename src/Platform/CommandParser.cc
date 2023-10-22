#include "CommandParser.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "Colors.h"
#include "Utils.h"

namespace platform {
    void CommandParser::messageError(const string& message)
    {
        cout << Colors::RED() << message << Colors::RESET() << endl;
    }
    pair<char, int> CommandParser::parse(const string& color, const vector<tuple<string, char, bool>>& options, const char defaultCommand, const int maxIndex)
    {
        bool finished = false;
        while (!finished) {
            stringstream oss;
            string line;
            oss << color << "Choose option (";
            bool first = true;
            for (auto& option : options) {
                if (first) {
                    first = false;
                } else {
                    oss << ", ";
                }
                oss << get<char>(option) << "=" << get<string>(option);
            }
            oss << "): ";
            cout << oss.str();
            getline(cin, line);
            cout << Colors::RESET();
            line = trim(line);
            if (line.size() == 0)
                continue;
            if (all_of(line.begin(), line.end(), ::isdigit)) {
                command = defaultCommand;
                index = stoi(line);
                if (index > maxIndex || index < 0) {
                    messageError("Index out of range");
                    continue;
                }
                finished = true;
                break;
            }
            bool found = false;
            for (auto& option : options) {
                if (line[0] == get<char>(option)) {
                    found = true;
                    // it's a match
                    line.erase(line.begin());
                    line = trim(line);
                    if (get<bool>(option)) {
                        // The option requires a value
                        if (line.size() == 0) {
                            messageError("Option " + get<string>(option) + " requires a value");
                            break;
                        }
                        try {
                            index = stoi(line);
                            if (index > maxIndex || index < 0) {
                                messageError("Index out of range");
                                break;
                            }
                        }
                        catch (const std::invalid_argument& ia) {
                            messageError("Invalid value: " + line);
                            break;
                        }
                    } else {
                        if (line.size() > 0) {
                            messageError("option " + get<string>(option) + " doesn't accept values");
                            break;
                        }
                    }
                    command = get<char>(option);
                    finished = true;
                    break;
                }
            }
            if (!found) {
                messageError("I don't know " + line);
            }
        }
        return { command, index };
    }
} /* namespace platform */