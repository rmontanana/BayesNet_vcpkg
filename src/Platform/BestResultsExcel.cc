#include "BestResultsExcel.h"
#include "Paths.h"

namespace platform {
    BestResultsExcel::BestResultsExcel(vector<string> models, vector<string> datasets, json table, bool friedman) : models(models), datasets(datasets), table(table), friedman(friedman)
    {
        workbook = workbook_new((Paths::excel() + fileName).c_str());
        worksheet = workbook_add_worksheet(workbook, "Best Results");
        setProperties("Best Results");
        createFormats();
        formatColumns();
    }

    BestResultsExcel::~BestResultsExcel()
    {
        workbook_close(workbook);
    }
    void BestResultsExcel::formatColumns()
    {

    }
    void BestResultsExcel::build()
    {
        header();
        body();
        footer();
    }
    void BestResultsExcel::header()
    {

    }
    void BestResultsExcel::body()
    {

    }
    void BestResultsExcel::footer()
    {

    }

}