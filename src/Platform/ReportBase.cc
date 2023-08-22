#include <sstream>
#include <locale>
#include "ReportBase.h"
#include "BestResult.h"


namespace platform {
    string ReportBase::fromVector(const string& key)
    {
        stringstream oss;
        string sep = "";
        oss << "[";
        for (auto& item : data[key]) {
            oss << sep << item.get<double>();
            sep = ", ";
        }
        oss << "]";
        return oss.str();
    }
    string ReportBase::fVector(const string& title, const json& data, const int width, const int precision)
    {
        stringstream oss;
        string sep = "";
        oss << title << "[";
        for (const auto& item : data) {
            oss << sep << fixed << setw(width) << setprecision(precision) << item.get<double>();
            sep = ", ";
        }
        oss << "]";
        return oss.str();
    }
    void ReportBase::show()
    {
        header();
        body();
    }
}