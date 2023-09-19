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

    lxw_format* ReportExcel::efectiveStyle(const string& style)
    {
        lxw_format* efectiveStyle;
        if (style == "") {
            efectiveStyle = NULL;
        } else {
            string suffix = row % 2 ? "_odd" : "_even";
            efectiveStyle = styles.at(style + suffix);
        }
        return efectiveStyle;
    }

    void ReportExcel::writeString(int row, int col, const string& text, const string& style)
    {
        worksheet_write_string(worksheet, row, col, text.c_str(), efectiveStyle(style));
    }
    void ReportExcel::writeInt(int row, int col, const int number, const string& style)
    {
        worksheet_write_number(worksheet, row, col, number, efectiveStyle(style));
    }
    void ReportExcel::writeDouble(int row, int col, const double number, const string& style)
    {
        worksheet_write_number(worksheet, row, col, number, efectiveStyle(style));
    }

    void ReportExcel::formatColumns()
    {
        worksheet_freeze_panes(worksheet, 6, 1);
        vector<int> columns_sizes = { 22, 10, 9, 7, 12, 12, 12, 12, 12, 3, 14, 12, 50 };
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }

    void ReportExcel::addColor(lxw_format* style, bool odd)
    {
        uint32_t efectiveColor = odd ? colorEven : colorOdd;
        format_set_bg_color(style, lxw_color_t(efectiveColor));
    }
    void ReportExcel::createStyle(const string& name, lxw_format* style, bool odd)
    {
        addColor(style, odd);
        if (name == "textCentered") {
            format_set_align(style, LXW_ALIGN_CENTER);
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "text") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "bodyHeader") {
            format_set_bold(style);
            format_set_font_size(style, normalSize);
            format_set_align(style, LXW_ALIGN_VERTICAL_CENTER);
            format_set_align(style, LXW_ALIGN_CENTER);
            format_set_align(style, LXW_ALIGN_VERTICAL_CENTER);
            format_set_bg_color(style, lxw_color_t(colorTitle));
        } else if (name == "result") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
            format_set_num_format(style, "0.0000000");
        } else if (name == "time") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
            format_set_num_format(style, "#,##0.000000");
        } else if (name == "ints") {
            format_set_font_size(style, normalSize);
            format_set_num_format(style, "###,###");
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "floats") {
            format_set_border(style, LXW_BORDER_THIN);
            format_set_font_size(style, normalSize);
            format_set_num_format(style, "#,###.00");
        }
    }

    void ReportExcel::createFormats()
    {
        auto styleNames = { "text", "textCentered", "bodyHeader", "result", "time", "ints", "floats" };
        lxw_format* style;
        for (string name : styleNames) {
            lxw_format* style = workbook_add_format(workbook);
            style = workbook_add_format(workbook);
            createStyle(name, style, true);
            styles[name + "_odd"] = style;
            style = workbook_add_format(workbook);
            createStyle(name, style, false);
            styles[name + "_even"] = style;
        }

        // Header 1st line
        lxw_format* headerFirst = workbook_add_format(workbook);
        format_set_bold(headerFirst);
        format_set_font_size(headerFirst, 18);
        format_set_align(headerFirst, LXW_ALIGN_CENTER);
        format_set_align(headerFirst, LXW_ALIGN_VERTICAL_CENTER);
        format_set_border(headerFirst, LXW_BORDER_THIN);
        format_set_bg_color(headerFirst, lxw_color_t(colorTitle));

        // Header rest
        lxw_format* headerRest = workbook_add_format(workbook);
        format_set_bold(headerRest);
        format_set_align(headerRest, LXW_ALIGN_CENTER);
        format_set_font_size(headerRest, 16);
        format_set_align(headerRest, LXW_ALIGN_VERTICAL_CENTER);
        format_set_border(headerRest, LXW_BORDER_THIN);
        format_set_bg_color(headerRest, lxw_color_t(colorOdd));

        // Header small
        lxw_format* headerSmall = workbook_add_format(workbook);
        format_set_bold(headerSmall);
        format_set_align(headerSmall, LXW_ALIGN_LEFT);
        format_set_font_size(headerSmall, 12);
        format_set_border(headerSmall, LXW_BORDER_THIN);
        format_set_align(headerSmall, LXW_ALIGN_VERTICAL_CENTER);
        format_set_bg_color(headerSmall, lxw_color_t(colorOdd));

        styles["headerFirst"] = headerFirst;
        styles["headerRest"] = headerRest;
        styles["headerSmall"] = headerSmall;
    }

    void ReportExcel::setProperties()
    {
        char line[data["title"].get<string>().size() + 1];
        strcpy(line, data["title"].get<string>().c_str());
        /* Create a properties structure and set some of the fields. */
        lxw_doc_properties properties = {
            .title = line,
            .subject = "Machine learning results",
            .author = "Ricardo Montañana Gómez",
            .manager = "Dr. J. A. Gámez, Dr. J. M. Puerta",
            .company = "UCLM",
            .comments = "Created with libxlsxwriter and c++",
        };

        /* Set the properties in the workbook. */
        workbook_set_properties(workbook, &properties);
    }

    void ReportExcel::createFile()
    {
        workbook = workbook_new((Paths::excel() + fileName).c_str());
        const string name = data["model"].get<string>();
        worksheet = workbook_add_worksheet(workbook, name.c_str());
        setProperties();
        createFormats();
        formatColumns();
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
        string message = data["model"].get<string>() + " ver. " + data["version"].get<string>() +
            data["language"].get<string>() + " ver. " + data["language_version"].get<string>() +
            " with " + to_string(data["folds"].get<int>()) + " Folds cross validation and " + to_string(data["seeds"].size()) +
            " random seeds. " + data["date"].get<string>() + " " + data["time"].get<string>();
        worksheet_merge_range(worksheet, 0, 0, 0, 12, message.c_str(), styles["headerFirst"]);
        worksheet_merge_range(worksheet, 1, 0, 1, 12, data["title"].get<string>().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 0, 3, 0, ("Score is " + data["score_name"].get<string>()).c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 1, 3, 3, "Execution time", styles["headerRest"]);
        oss << setprecision(2) << fixed << data["duration"].get<float>() << " s";
        worksheet_merge_range(worksheet, 2, 4, 2, 5, oss.str().c_str(), styles["headerRest"]);
        oss.str("");
        oss.clear();
        oss << setprecision(2) << fixed << data["duration"].get<float>() / 3600 << " h";
        worksheet_merge_range(worksheet, 3, 4, 3, 5, oss.str().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 6, 3, 7, "Platform", styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 8, 3, 9, data["platform"].get<string>().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 10, 2, 12, ("Random seeds: " + fromVector("seeds")).c_str(), styles["headerSmall"]);
        oss.str("");
        oss.clear();
        oss << "Stratified: " << (data["stratified"].get<bool>() ? "True" : "False");
        worksheet_merge_range(worksheet, 3, 10, 3, 11, oss.str().c_str(), styles["headerSmall"]);
        oss.str("");
        oss.clear();
        oss << "Discretized: " << (data["discretized"].get<bool>() ? "True" : "False");
        worksheet_write_string(worksheet, 3, 12, oss.str().c_str(), styles["headerSmall"]);
    }

    void ReportExcel::body()
    {
        auto head = vector<string>(
            { "Dataset", "Samples", "Features", "Classes", "Nodes", "Edges", "States", "Score", "Score Std.", "St.", "Time",
             "Time Std.", "Hyperparameters" });
        int col = 0;
        for (const auto& item : head) {
            writeString(5, col++, item, "bodyHeader");
        }
        row = 6;
        col = 0;
        json lastResult;
        double totalScore = 0.0;
        string hyperparameters;
        for (const auto& r : data["results"]) {
            writeString(row, col, r["dataset"].get<string>(), "text");
            writeInt(row, col + 1, r["samples"].get<int>(), "ints");
            writeInt(row, col + 2, r["features"].get<int>(), "ints");
            writeInt(row, col + 3, r["classes"].get<int>(), "ints");
            writeDouble(row, col + 4, r["nodes"].get<float>(), "floats");
            writeDouble(row, col + 5, r["leaves"].get<float>(), "floats");
            writeDouble(row, col + 6, r["depth"].get<double>(), "floats");
            writeDouble(row, col + 7, r["score"].get<double>(), "result");
            writeDouble(row, col + 8, r["score_std"].get<double>(), "result");
            const string status = "X";
            writeString(row, col + 9, status, "textCentered");
            writeDouble(row, col + 10, r["time"].get<double>(), "time");
            writeDouble(row, col + 11, r["time_std"].get<double>(), "time");
            try {
                hyperparameters = r["hyperparameters"].get<string>();
            }
            catch (const exception& err) {
                stringstream oss;
                oss << r["hyperparameters"];
                hyperparameters = oss.str();
            }
            writeString(row, col + 12, hyperparameters, "text");
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
    }

    void ReportExcel::footer(double totalScore, int row)
    {
        auto score = data["score_name"].get<string>();
        if (score == BestResult::scoreName()) {
            worksheet_merge_range(worksheet, row + 2, 1, row + 2, 5, (score + " compared to " + BestResult::title() + " .:").c_str(), styles["text_even"]);
            writeDouble(row + 2, 6, totalScore / BestResult::score(), "result");
        }
    }
}