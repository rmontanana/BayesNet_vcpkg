#include <sstream>
#include <locale>
#include "Datasets.h"
#include "ReportBase.h"
#include "BestResult.h"


namespace platform {
    ReportBase::ReportBase(json data_) : margin(0.1), data(data_)
    {
        stringstream oss;
        oss << "Better than ZeroR + " << setprecision(1) << fixed << margin * 100 << "%";
        meaning = {
            {Symbols::equal_best, "Equal to best"},
            {Symbols::better_best, "Better than best"},
            {Symbols::cross, "Less than or equal to ZeroR"},
            {Symbols::upward_arrow, oss.str()}
        };
    }
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
    string ReportBase::compareResult(const string& dataset, double result)
    {
        string status = " ";
        if (data["score_name"].get<string>() == "accuracy") {
            auto dt = Datasets(Paths::datasets(), false);
            dt.loadDataset(dataset);
            auto numClasses = dt.getNClasses(dataset);
            if (numClasses == 2) {
                vector<int> distribution = dt.getClassesCounts(dataset);
                double nSamples = dt.getNSamples(dataset);
                vector<int>::iterator maxValue = max_element(distribution.begin(), distribution.end());
                double mark = *maxValue / nSamples * (1 + margin);
                if (mark > 1) {
                    mark = 0.9995;
                }
                status = result < mark ? Symbols::cross : result > mark ? Symbols::upward_arrow : "=";
                auto item = summary.find(status);
                if (item != summary.end()) {
                    summary[status]++;
                } else {
                    summary[status] = 1;
                }
            }
        }
        return status;
    }
}