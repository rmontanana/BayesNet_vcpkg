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
    protected:
        void writeString(int row, int col, const string& text, const string& style = "");
        void writeInt(int row, int col, const int number, const string& style = "");
        void writeDouble(int row, int col, const double number, const string& style = "");
        void formatHeader();
        void formatBody();
        void formatFooter();
        void createFormats();
    private: galeote
        void createFile();
           void closeFile();
           lxw_workbook* workbook;
           lxw_worksheet* worksheet;
           map<string, lxw_format*> styles;
           void header() override;
           void body() override;
           void footer(double totalScore, int row);
    };
};
#endif // !REPORTEXCEL_H