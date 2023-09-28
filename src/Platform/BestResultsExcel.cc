#include <sstream>
#include "BestResultsExcel.h"
#include "Paths.h"
#include "Statistics.h"

namespace platform {
    BestResultsExcel::BestResultsExcel(string score, vector<string> models, vector<string> datasets, json table, bool friedman, double significance) : score(score), models(models), datasets(datasets), table(table), friedman(friedman), significance(significance)
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
            worksheet = workbook_add_worksheet(workbook, "Friedman");
            vector<int> columns_sizes = { 5, datasetNameSize };
            for (int i = 0; i < models.size(); ++i) {
                columns_sizes.push_back(modelNameSize);
            }
            for (int i = 0; i < columns_sizes.size(); ++i) {
                worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
            }
            worksheet_merge_range(worksheet, 0, 0, 0, 1 + models.size(), "Friedman Test", styles["headerFirst"]);
            row = 2;
            Statistics stats(models, datasets, table, significance, false);
            auto result = stats.friedmanTest();
            stats.postHocHolmTest(result);
            auto friedmanResult = stats.getFriedmanResult();
            auto holmResult = stats.getHolmResult();
            worksheet_merge_range(worksheet, row, 0, row, 1 + models.size(), "Null hypothesis: H0 'There is no significant differences between all the classifiers.'", styles["headerSmall"]);
            row += 2;
            writeString(row, 1, "Friedman Q", "bodyHeader");
            writeDouble(row, 2, friedmanResult.statistic, "bodyHeader");
            row++;
            writeString(row, 1, "Critical χ2 value", "bodyHeader");
            writeDouble(row, 2, friedmanResult.criticalValue, "bodyHeader");
            row++;
            writeString(row, 1, "p-value", "bodyHeader");
            writeDouble(row, 2, friedmanResult.pvalue, "bodyHeader");
            writeString(row, 3, friedmanResult.reject ? "<" : ">", "bodyHeader");
            writeDouble(row, 4, significance, "bodyHeader");
            writeString(row, 5, friedmanResult.reject ? "Reject H0" : "Accept H0", "bodyHeader");
            row += 3;
            worksheet_merge_range(worksheet, row, 0, row, 1 + models.size(), "Holm Test", styles["headerFirst"]);
            row += 2;
            worksheet_merge_range(worksheet, row, 0, row, 1 + models.size(), "Null hypothesis: H0 'There is no significant differences between the control model and the other models.'", styles["headerSmall"]);
            row += 2;
            string controlModel = "Control Model: " + holmResult.model;
            worksheet_merge_range(worksheet, row, 1, row, 7, controlModel.c_str(), styles["bodyHeader_odd"]);
            row++;
            writeString(row, 1, "Model", "bodyHeader");
            writeString(row, 2, "p-value", "bodyHeader");
            writeString(row, 3, "Rank", "bodyHeader");
            writeString(row, 4, "Win", "bodyHeader");
            writeString(row, 5, "Tie", "bodyHeader");
            writeString(row, 6, "Loss", "bodyHeader");
            writeString(row, 7, "Reject H0", "bodyHeader");
            row++;
            for (const auto& item : holmResult.holmLines) {
                writeString(row, 1, item.model, "text");
                writeDouble(row, 2, item.pvalue, "result");
                writeDouble(row, 3, item.rank, "result");
                writeInt(row, 4, item.wtl.win, "ints");
                writeInt(row, 5, item.wtl.tie, "ints");
                writeInt(row, 6, item.wtl.loss, "ints");
                writeString(row, 7, item.reject ? "Yes" : "No", "textCentered");
                row++;
            }
        }
    }
}