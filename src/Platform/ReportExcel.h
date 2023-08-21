#ifndef REPORTEXCEL_H
#define REPORTEXCEL_H
#include "ReportBase.h"
#include "Colors.h"
namespace platform {
    using namespace std;
    const int MAXLL = 128;
    class ReportExcel : public ReportBase{
    public:
        explicit ReportExcel(json data_) : ReportBase(data_) {};
        virtual ~ReportExcel() = default;
    private:
        void header() override;
        void body() override;
        void footer() override;
    };
};
#endif // !REPORTEXCEL_H