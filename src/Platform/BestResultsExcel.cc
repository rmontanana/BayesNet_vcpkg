#include <sstream>
#include "BestResultsExcel.h"
#include "Paths.h"
#include <iostream>

namespace platform {
    BestResultsExcel::BestResultsExcel(string score, vector<string> models, vector<string> datasets, json table, bool friedman) : score(score), models(models), datasets(datasets), table(table), friedman(friedman)
    {
        workbook = workbook_new((Paths::excel() + fileName).c_str());
        worksheet = workbook_add_worksheet(workbook, "Best Results");
        setProperties("Best Results");
        createFormats();
        int maxModelName = (*max_element(models.begin(), models.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        modelNameSize = max(modelNameSize, maxModelName);
        int maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        datasetNameSize = max(datasetNameSize, maxDatasetName);
        formatColumns();
    }
    BestResultsExcel::~BestResultsExcel()
    {
        workbook_close(workbook);
    }
    void BestResultsExcel::formatColumns()
    {
        worksheet_freeze_panes(worksheet, 4, 2);
        vector<int> columns_sizes = { 5, datasetNameSize };
        for (int i = 0; i < models.size(); ++i) {
            columns_sizes.push_back(modelNameSize);
        }
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }
    void BestResultsExcel::build()
    {
        header();
        body();
        footer();
    }
    void BestResultsExcel::header()
    {
        row = 0;
        string message = "Best results for " + score;
        worksheet_merge_range(worksheet, 0, 0, 0, 1 + models.size(), message.c_str(), styles["headerFirst"]);

        // Body header
        row = 3;
        int col = 1;
        writeString(row, 0, "Nº", "bodyHeader");
        writeString(row, 1, "Dataset", "bodyHeader");
        for (const auto& model : models) {
            writeString(row, ++col, model.c_str(), "bodyHeader");
        }
    }
    void BestResultsExcel::body()
    {
        row = 4;
        int i = 0;
        json origin = table.begin().value();
        for (auto const& item : origin.items()) {
            writeInt(row, 0, i++, "ints");
            writeString(row, 1, item.key().c_str(), "text");
            int col = 1;
            for (const auto& model : models) {
                double value = table[model].at(item.key()).at(0).get<double>();
                writeDouble(row, ++col, value, "result");
            }
            ++row;
        }
        // Set Totals
        writeString(row, 1, "Total", "bodyHeader");
        int col = 1;
        for (const auto& model : models) {
            stringstream oss;
            oss << "=sum(indirect(address(" << 5 << "," << col + 2 << ")):indirect(address(" << row << "," << col + 2 << ")))";
            worksheet_write_formula(worksheet, row, ++col, oss.str().c_str(), styles["bodyHeader_odd"]);
        }
    }
    void BestResultsExcel::footer()
    {
        if (friedman) {

        }
    }
}