#ifndef REPORTBASE_H
#define REPORTBASE_H
#include <string>
#include <iostream>
#include "Paths.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace platform {
    using namespace std;
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
    class ReportBase {
    public:
        explicit ReportBase(json data_);
        virtual ~ReportBase() = default;
        void show();
    protected:
        json data;
        string fromVector(const string& key);
        string fVector(const string& title, const json& data, const int width, const int precision);
        virtual void header() = 0;
        virtual void body() = 0;
        virtual void showSummary() = 0;
        string compareResult(const string& dataset, double result);
        map<string, int> summary;
        double margin;
        map<string, string> meaning;
    };
};
#endif