#ifndef REPORTBASE_H
#define REPORTBASE_H
#include <string>
#include <iostream>
#include "Paths.h"
#include "Symbols.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace platform {
    using namespace std;

    class ReportBase {
    public:
        explicit ReportBase(json data_, bool compare);
        virtual ~ReportBase() = default;
        void show();
    protected:
        json data;
        string fromVector(const string& key);
        string fVector(const string& title, const json& data, const int width, const int precision);
        bool getExistBestFile();
        virtual void header() = 0;
        virtual void body() = 0;
        virtual void showSummary() = 0;
        string compareResult(const string& dataset, double result);
        map<string, int> summary;
        double margin;
        map<string, string> meaning;
        bool compare;
    private:
        double bestResult(const string& dataset, const string& model);
        json bestResults;
        bool existBestFile = true;
    };
};
#endif