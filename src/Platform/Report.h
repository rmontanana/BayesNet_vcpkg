#ifndef REPORT_H
#define REPORT_H
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
const int MAXL = 121;
namespace platform {
    using namespace std;
    class Report {
    public:
        explicit Report(json data_) { data = data_; };
        virtual ~Report() = default;
        void show();
    private:
        void header();
        void body();
        string fromVector(const string& key);
        json data;
    };
};
#endif