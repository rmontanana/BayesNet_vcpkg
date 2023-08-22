#ifndef REPORTEXCEL_H
#define REPORTEXCEL_H
#include <OpenXLSX.hpp>
#include "ReportBase.h"
#include "Paths.h"
#include "Colors.h"
namespace platform {
    using namespace std;
    using namespace OpenXLSX;
    const int MAXLL = 128;
    class ReportExcel : public ReportBase{
    public:
        explicit ReportExcel(json data_) : ReportBase(data_) {createFile();};
        virtual ~ReportExcel() {closeFile();};
    private:
        void createFile();
        void closeFile();
        XLDocument doc;
        XLWorksheet wks;
        void header() override;
        void body() override;
        void footer(double totalScore, int row);
    };
};
#endif // !REPORTEXCEL_H