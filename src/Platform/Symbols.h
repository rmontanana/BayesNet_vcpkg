#ifndef SYMBOLS_H
#define SYMBOLS_H
#include <string>
using namespace std;
namespace platform {
    class Symbols {
    public:
        inline static const string check_mark{ "\u2714" };
        inline static const string exclamation{ "\u2757" };
        inline static const string black_star{ "\u2605" };
        inline static const string cross{ "\u2717" };
        inline static const string upward_arrow{ "\u27B6" };
        inline static const string down_arrow{ "\u27B4" };
        inline static const string equal_best{ check_mark };
        inline static const string better_best{ black_star };
    };
}
#endif // !SYMBOLS_H