#ifndef REPORTCONSOLE_H
#define REPORTCONSOLE_H
#include <string>
#include <iostream>
#include "ReportBase.h"
#include "Colors.h"

namespace platform {
    using namespace std;
    const int MAXL = 128;
    class ReportConsole : public ReportBase{
    public:
        explicit ReportConsole(json data_) : ReportBase(data_) {};
        virtual ~ReportConsole() = default;
    private:
        string headerLine(const string& text);
        void header() override;
        void body() override;
        void footer(double totalScore);
    };
};
#endif