#ifndef REPORTEXCEL_H
#define REPORTEXCEL_H
#include<map>
#include "xlsxwriter.h"
#include "ReportBase.h"
#include "Colors.h"
namespace platform {
    using namespace std;
    const int MAXLL = 128;

    class ReportExcel : public ReportBase {
    public:
        explicit ReportExcel(json data_, bool compare, lxw_workbook* workbook);
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
        lxw_workbook* workbook;
        lxw_worksheet* worksheet;
        map<string, lxw_format*> styles;
        int row;
        int normalSize; //font size for report body
        uint32_t colorTitle;
        uint32_t colorOdd;
        uint32_t colorEven;
        const string fileName = "some_results.xlsx";
        void header() override;
        void body() override;
        void showSummary() override;
        void footer(double totalScore, int row);
        void createStyle(const string& name, lxw_format* style, bool odd);
        void addColor(lxw_format* style, bool odd);
        lxw_format* efectiveStyle(const string& name);
    };
};
#endif // !REPORTEXCEL_H