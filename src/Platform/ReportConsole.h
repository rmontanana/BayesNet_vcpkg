#ifndef REPORTCONSOLE_H
#define REPORTCONSOLE_H
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include "ReportBase.h"
#include "Colors.h"

using json = nlohmann::json;
const int MAXL = 128;
namespace platform {
    using namespace std;
    class ReportConsole : public ReportBase{
    public:
        explicit ReportConsole(json data_) : ReportBase(data_) {};
        virtual ~ReportConsole() = default;
    private:
    
        void header() override;
        void body() override;
        void footer() override;
    };
};
#endif