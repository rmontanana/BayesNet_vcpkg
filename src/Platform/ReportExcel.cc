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
    void ReportExcel::writeString(int row, int col, const string& text, const string& style)
    {
        lxw_format* efectiveStyle = style == "" ? NULL : styles[style];
        worksheet_write_string(worksheet, row, col, text.c_str(), efectiveStyle);
    }
    void ReportExcel::writeInt(int row, int col, const int number, const string& style)
    {
        lxw_format* efectiveStyle = style == "" ? NULL : styles[style];
        worksheet_write_number(worksheet, row, col, number, efectiveStyle);
    }
    void ReportExcel::writeDouble(int row, int col, const double number, const string& style)
    {
        lxw_format* efectiveStyle = style == "" ? NULL : styles[style];
        worksheet_write_number(worksheet, row, col, number, efectiveStyle);
    }

    void ReportExcel::formatHeader()
    {
        worksheet_freeze_panes(worksheet, 8, 0);
    }

    void ReportExcel::formatBody()
    {

    }

    void ReportExcel::formatFooter()
    {

    }

    void ReportExcel::createFormats()
    {
        lxw_format* bold = workbook_add_format(workbook);
        format_set_bold(bold);

        lxw_format* result = workbook_add_format(workbook);
        format_set_num_format(result, "0.0000000");

        lxw_format* timeStyle = workbook_add_format(workbook);
        format_set_num_format(timeStyle, "#,##0.00");

        lxw_format* ints = workbook_add_format(workbook);
        format_set_num_format(ints, "###,###");

        lxw_format* floats = workbook_add_format(workbook);
        format_set_num_format(floats, "#,###.00");

        styles["bold"] = bold;
        styles["result"] = result;
        styles["time"] = timeStyle;
        styles["ints"] = ints;
        styles["floats"] = floats
    }

    void ReportExcel::createFile()
    {
        workbook = workbook_new((Paths::excel() + "some_results.xlsx").c_str());
        const string name = data["model"].get<string>();
        worksheet = workbook_add_worksheet(workbook, name.c_str());
        createFormats();
    }

    void ReportExcel::closeFile()
    {
        workbook_close(workbook);
    }

    void ReportExcel::header()
    {
        locale mylocale(cout.getloc(), new separated);
        locale::global(mylocale);
        cout.imbue(mylocale);
        stringstream oss;
        writeString(0, 0, "Report " + data["model"].get<string>() + " ver. " + data["version"].get<string>() + " with " +
            to_string(data["folds"].get<int>()) + " Folds cross validation and " + to_string(data["seeds"].size()) +
            " random seeds. " + data["date"].get<string>() + " " + data["time"].get<string>(), "bold");
        writeString(1, 0, data["title"].get<string>());
        writeString(2, 0, "Random seeds: " + fromVector("seeds") + " Stratified: " +
            (data["stratified"].get<bool>() ? "True" : "False"));
        oss << "Execution took  " << setprecision(2) << fixed << data["duration"].get<float>() << " seconds,   "
            << data["duration"].get<float>() / 3600 << " hours, on " << data["platform"].get<string>();
        writeString(3, 0, oss.str());
        writeString(4, 0, "Score is " + data["score_name"].get<string>());
        formatHeader();
    }

    void ReportExcel::body()
    {
        auto head = vector<string>(
            { "Dataset", "Samples", "Features", "Classes", "Nodes", "Edges", "States", "Score", "Score Std.", "Time",
             "Time Std.", "Hyperparameters" });
        int col = 1;
        for (const auto& item : head) {
            writeString(8, col++, item);
        }
        int row = 9;
        col = 1;
        json lastResult;
        double totalScore = 0.0;
        string hyperparameters;
        for (const auto& r : data["results"]) {
            writeString(row, col, r["dataset"].get<string>());
            writeInt(row, col + 1, r["samples"].get<int>(), "ints");
            writeInt(row, col + 2, r["features"].get<int>(), "ints");
            writeInt(row, col + 3, r["classes"].get<int>(), "ints");
            writeDouble(row, col + 4, r["nodes"].get<float>(), "floats");
            writeDouble(row, col + 5, r["leaves"].get<float>(), "floats");
            writeDouble(row, col + 6, r["depth"].get<double>(), "floats");
            writeDouble(row, col + 7, r["score"].get<double>(), "result");
            writeDouble(row, col + 8, r["score_std"].get<double>(), "result");
            writeDouble(row, col + 9, r["time"].get<double>(), "time");
            writeDouble(row, col + 10, r["time_std"].get<double>(), "time");
            try {
                hyperparameters = r["hyperparameters"].get<string>();
            }
            catch (const exception& err) {
                stringstream oss;
                oss << r["hyperparameters"];
                hyperparameters = oss.str();
            }
            writeString(row, col + 11, hyperparameters);
            lastResult = r;
            totalScore += r["score"].get<double>();
            row++;
        }
        if (data["results"].size() == 1) {
            for (const string& group : { "scores_train", "scores_test", "times_train", "times_test" }) {
                row++;
                col = 1;
                writeString(row, col, group);
                for (double item : lastResult[group]) {
                    writeDouble(row, ++col, item);
                }
            }
        } else {
            footer(totalScore, row);
        }
        formatBody();
    }

    void ReportExcel::footer(double totalScore, int row)
    {
        auto score = data["score_name"].get<string>();
        if (score == BestResult::scoreName()) {
            writeString(row + 2, 1, score + " compared to " + BestResult::title() + " .:  ");
            writeDouble(row + 2, 7, totalScore / BestResult::score(), "result");
        }
        formatFooter();
    }
}