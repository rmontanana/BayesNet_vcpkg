#include <sstream>
#include <locale>
#include "Datasets.h"
#include "ReportBase.h"
#include "BestResult.h"


namespace platform {
    ReportBase::ReportBase(json data_, bool compare) : data(data_), compare(compare), margin(0.1)
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
        if (compare) {
            double best = bestResult(dataset, data["model"].get<string>());
            if (result == best) {
                status = Symbols::equal_best;
            } else if (result > best) {
                status = Symbols::better_best;
            }
        } else {
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
                }
            }
        }
        if (status != " ") {
            auto item = summary.find(status);
            if (item != summary.end()) {
                summary[status]++;
            } else {
                summary[status] = 1;
            }
        }
        return status;
    }
    double ReportBase::bestResult(const string& dataset, const string& model)
    {
        double value = 0.0;
        if (bestResults.size() == 0) {
            // try to load the best results
            string score = data["score_name"];
            replace(score.begin(), score.end(), '_', '-');
            string fileName = "best_results_" + score + "_" + model + ".json";
            ifstream resultData(Paths::results() + "/" + fileName);
            if (resultData.is_open()) {
                bestResults = json::parse(resultData);
            }
        }
        try {
            value = bestResults.at(dataset).at(0);
        }
        catch (exception) {
            value = 1.0;
        }
        return value;
    }
}