#ifndef REPORTEXCEL_H
#define REPORTEXCEL_H
#include<map>
#include "xlsxwriter.h"
#include "ReportBase.h"
#include "Paths.h"
#include "Colors.h"
namespace platform {
    using namespace std;
    const int MAXLL = 128;
    class Symbols {
    public:
        inline static const string check_mark{ "\u2714" };
        inline static const string exclamation{ "\u2757" };
        inline static const string black_star{ "\u2605" };
        inline static const string cross{ "\u2717" };
        inline static const string upward_arrow{ "\u27B6" };
        inline static const string down_arrow{ "\u27B4" };
        inline static const string equal_best{ check_mark };
        inline static const string better_best{ black_star };
    };
    class ReportExcel : public ReportBase {
    public:
        explicit ReportExcel(json data_, lxw_workbook* workbook);
        lxw_workbook* getWorkbook();
    private:
        void writeString(int row, int col, const string& text, const string& style = "");
        void writeInt(int row, int col, const int number, const string& style = "");
        void writeDouble(int row, int col, const double number, const string& style = "");
        void formatColumns();
        void createFormats();
        void setProperties();
        void createFile();
        void closeFile();
        void showSummary();
        lxw_workbook* workbook;
        lxw_worksheet* worksheet;
        map<string, lxw_format*> styles;
        map<string, int> summary;
        int row;
        int normalSize; //font size for report body
        uint32_t colorTitle;
        uint32_t colorOdd;
        uint32_t colorEven;
        double margin;
        const string fileName = "some_results.xlsx";
        void header() override;
        void body() override;
        void footer(double totalScore, int row);
        void createStyle(const string& name, lxw_format* style, bool odd);
        void addColor(lxw_format* style, bool odd);
        lxw_format* efectiveStyle(const string& name);
        string compareResult(const string& dataset, double result);
    };
};
#endif // !REPORTEXCEL_H