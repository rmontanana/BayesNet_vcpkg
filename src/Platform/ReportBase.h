#ifndef REPORTBASE_H
#define REPORTBASE_H
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace platform {
    using namespace std;
    class ReportBase {
    public:
        explicit ReportBase(json data_) { data = data_; };
        virtual ~ReportBase() = default;
        void show();
    protected:
        json data;
        string fromVector(const string& key);
        string fVector(const string& title, const json& data, const int width, const int precision);
        virtual void header() = 0;
        virtual void body() = 0;
    };
};
#endif