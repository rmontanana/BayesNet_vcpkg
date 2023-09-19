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
    class ReportExcel : public ReportBase {
    public:
        explicit ReportExcel(json data_) : ReportBase(data_) { createFile(); };
        virtual ~ReportExcel() { closeFile(); };
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
        int row = 0;
        int normalSize = 14; //font size for report body
        uint32_t colorTitle = 0xB1A0C7;
        uint32_t colorOdd = 0xDCE6F1;
        uint32_t colorEven = 0xFDE9D9;
        const string fileName = "some_results.xlsx";
        void header() override;
        void body() override;
        void footer(double totalScore, int row);
        void createStyle(const string& name, lxw_format* style, bool odd);
        void addColor(lxw_format* style, bool odd);
        lxw_format* efectiveStyle(const string& name);
    };
};
#endif // !REPORTEXCEL_H