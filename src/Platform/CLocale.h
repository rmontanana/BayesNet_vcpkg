#ifndef LOCALE_H
#define LOCALE_H
#include <locale>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
namespace platform {
    struct separation : numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        string do_grouping() const { return "\03"; }
    };
    class ConfigLocale {
    public:
        explicit ConfigLocale()
        {
            locale mylocale(cout.getloc(), new separation);
            locale::global(mylocale);
            cout.imbue(mylocale);
        }
    };
}
#endif 