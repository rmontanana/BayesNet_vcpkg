#include <sstream>
#include <locale>
#include "ReportExcel.h"
#include "BestResult.h"


namespace platform {
    struct separated : numpunct<char> {
        char do_decimal_point() const { return ','; }

        char do_thousands_sep() const { return '.'; }

        string do_grouping() const { return "\03"; }
    };

    void ReportExcel::createFile()
    {
        doc.create(Paths::excel() + "some_results.xlsx");
        wks = doc.workbook().worksheet("Sheet1");
        wks.setName(data["model"].get<string>());
    }

    void ReportExcel::closeFile()
    {
        doc.save();
        doc.close();
    }

    void ReportExcel::header()
    {
        locale mylocale(cout.getloc(), new separated);
        locale::global(mylocale);
        cout.imbue(mylocale);
        stringstream oss;
        wks.cell("A1").value().set(
            "Report " + data["model"].get<string>() + " ver. " + data["version"].get<string>() + " with " +
            to_string(data["folds"].get<int>()) + " Folds cross validation and " + to_string(data["seeds"].size()) +
            " random seeds. " + data["date"].get<string>() + " " + data["time"].get<string>());
        wks.cell("A2").value() = data["title"].get<string>();
        wks.cell("A3").value() = "Random seeds: " + fromVector("seeds") + " Stratified: " +
            (data["stratified"].get<bool>() ? "True" : "False");
        oss << "Execution took  " << setprecision(2) << fixed << data["duration"].get<float>() << " seconds,   "
            << data["duration"].get<float>() / 3600 << " hours, on " << data["platform"].get<string>();
        wks.cell("A4").value() = oss.str();
        wks.cell("A5").value() = "Score is " + data["score_name"].get<string>();
    }

    void ReportExcel::body()
    {
        auto header = vector<string>(
            { "Dataset", "Samples", "Features", "Classes", "Nodes", "Edges", "States", "Score", "Score Std.", "Time",
             "Time Std.", "Hyperparameters" });
        int col = 1;
        for (const auto& item : header) {
            wks.cell(8, col++).value() = item;
        }
        int row = 9;
        col = 1;
        json lastResult;
        double totalScore = 0.0;
        string hyperparameters;
        for (const auto& r : data["results"]) {
            wks.cell(row, col).value() = r["dataset"].get<string>();
            wks.cell(row, col + 1).value() = r["samples"].get<int>();
            wks.cell(row, col + 2).value() = r["features"].get<int>();
            wks.cell(row, col + 3).value() = r["classes"].get<int>();
            wks.cell(row, col + 4).value() = r["nodes"].get<float>();
            wks.cell(row, col + 5).value() = r["leaves"].get<float>();
            wks.cell(row, col + 6).value() = r["depth"].get<float>();
            wks.cell(row, col + 7).value() = r["score"].get<double>();
            wks.cell(row, col + 8).value() = r["score_std"].get<double>();
            wks.cell(row, col + 9).value() = r["time"].get<double>();
            wks.cell(row, col + 10).value() = r["time_std"].get<double>();
            try {
                hyperparameters = r["hyperparameters"].get<string>();
            }
            catch (const exception& err) {
                stringstream oss;
                oss << r["hyperparameters"];
                hyperparameters = oss.str();
            }
            wks.cell(row, col + 11).value() = hyperparameters;
            lastResult = r;
            totalScore += r["score"].get<double>();
            row++;
        }
        if (data["results"].size() == 1) {
            for (const string& group : { "scores_train", "scores_test", "times_train", "times_test" }) {
                row++;
                col = 1;
                wks.cell(row, col).value() = group;
                for (double item : lastResult[group]) {
                    wks.cell(row, ++col).value() = item;
                }
            }
        } else {
            footer(totalScore, row);
        }
    }

    void ReportExcel::footer(double totalScore, int row)
    {
        auto score = data["score_name"].get<string>();
        if (score == BestResult::scoreName()) {
            wks.cell(row + 2, 1).value() = score + " compared to " + BestResult::title() + " .:  ";
            wks.cell(row + 2, 5).value() = totalScore / BestResult::score();
        }
    }
}