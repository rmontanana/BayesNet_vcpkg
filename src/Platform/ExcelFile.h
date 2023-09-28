#ifndef EXCELFILE_H
#define EXCELFILE_H
#include <string>
#include <map>
#include "xlsxwriter.h"

using namespace std;
namespace platform {
    class ExcelFile {
    public:
        ExcelFile();
        ExcelFile(lxw_workbook* workbook);
        lxw_workbook* getWorkbook();
    protected:
        void setProperties(string title);
        void writeString(int row, int col, const string& text, const string& style = "");
        void writeInt(int row, int col, const int number, const string& style = "");
        void writeDouble(int row, int col, const double number, const string& style = "");
        void createFormats();
        void createStyle(const string& name, lxw_format* style, bool odd);
        void addColor(lxw_format* style, bool odd);
        lxw_format* efectiveStyle(const string& name);
        lxw_workbook* workbook;
        lxw_worksheet* worksheet;
        map<string, lxw_format*> styles;
        int row;
        int normalSize; //font size for report body
        uint32_t colorTitle;
        uint32_t colorOdd;
        uint32_t colorEven;
    private:
        void setDefault();
    };
}
#endif // !EXCELFILE_H