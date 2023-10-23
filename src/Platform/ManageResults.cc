#include "ManageResults.h"
#include "CommandParser.h"
#include <filesystem>
#include <tuple>
#include "Colors.h"
#include "CLocale.h"
#include "Paths.h"
#include "ReportConsole.h"
#include "ReportExcel.h"

namespace platform {

    ManageResults::ManageResults(int numFiles, const string& model, const string& score, bool complete, bool partial, bool compare) :
        numFiles{ numFiles }, complete{ complete }, partial{ partial }, compare{ compare }, results(Results(Paths::results(), model, score, complete, partial))
    {
        indexList = true;
        openExcel = false;
        workbook = NULL;
        if (numFiles == 0) {
            this->numFiles = results.size();
        }
    }
    void ManageResults::doMenu()
    {
        results.sortDate();
        list();
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        cout << Colors::RESET() << "Done!" << endl;
    }
    void ManageResults::list()
    {
        if (results.empty()) {
            cout << Colors::MAGENTA() << "No results found!" << Colors::RESET() << endl;
            exit(0);
        }
        auto temp = ConfigLocale();
        string suffix = numFiles != results.size() ? " of " + to_string(results.size()) : "";
        stringstream oss;
        oss << "Results on screen: " << numFiles << suffix;
        cout << Colors::GREEN() << oss.str() << endl;
        cout << string(oss.str().size(), '-') << endl;
        if (complete) {
            cout << Colors::MAGENTA() << "Only listing complete results" << endl;
        }
        if (partial) {
            cout << Colors::MAGENTA() << "Only listing partial results" << endl;
        }
        auto i = 0;
        int maxModel = results.maxModelSize();
        cout << Colors::GREEN() << " #  Date       " << setw(maxModel) << left << "Model" << " Score Name  Score       C / P Duration  Title" << endl;
        cout << "=== ========== " << string(maxModel, '=') << " =========== =========== === ========= =============================================================" << endl;
        bool odd = true;
        for (auto& result : results) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << result.to_string(maxModel) << endl;
            if (i == numFiles) {
                break;
            }
            odd = !odd;
        }
    }
    bool ManageResults::confirmAction(const string& intent, const string& fileName) const
    {
        string color;
        if (intent == "delete") {
            color = Colors::RED();
        } else {
            color = Colors::YELLOW();
        }
        string line;
        bool finished = false;
        while (!finished) {
            cout << color << "Really want to " << intent << " " << fileName << "? (y/n): ";
            getline(cin, line);
            finished = line.size() == 1 && (tolower(line[0]) == 'y' || tolower(line[0] == 'n'));
        }
        if (tolower(line[0]) == 'y') {
            return true;
        }
        cout << "Not done!" << endl;
        return false;
    }
    void ManageResults::report(const int index, const bool excelReport)
    {
        cout << Colors::YELLOW() << "Reporting " << results.at(index).getFilename() << endl;
        auto data = results.at(index).load();
        if (excelReport) {
            ReportExcel reporter(data, compare, workbook);
            reporter.show();
            openExcel = true;
            workbook = reporter.getWorkbook();
        } else {
            ReportConsole reporter(data, compare);
            reporter.show();
        }
    }
    void ManageResults::showIndex(const int index, const int idx)
    {
        // Show a dataset result inside a report
        auto data = results.at(index).load();
        cout << Colors::YELLOW() << "Showing " << results.at(index).getFilename() << endl;
        ReportConsole reporter(data, compare, idx);
        reporter.show();
    }
    void ManageResults::sortList()
    {
        cout << Colors::YELLOW() << "Choose sorting field (date='d', score='s', duration='u', model='m'): ";
        string line;
        char option;
        getline(cin, line);
        if (line.size() == 0)
            return;
        if (line.size() > 1) {
            cout << "Invalid option" << endl;
            return;
        }
        option = line[0];
        switch (option) {
            case 'd':
                results.sortDate();
                break;
            case 's':
                results.sortScore();
                break;
            case 'u':
                results.sortDuration();
                break;
            case 'm':
                results.sortModel();
                break;
            default:
                cout << "Invalid option" << endl;
        }
    }
    void ManageResults::menu()
    {
        char option;
        int index, subIndex;
        bool finished = false;
        string filename;
        // tuple<Option, digit, requires value>
        vector<tuple<string, char, bool>>  mainOptions = {
            {"quit", 'q', false},
            {"list", 'l', false},
            {"delete", 'd', true},
            {"hide", 'h', true},
            {"sort", 's', false},
            {"report", 'r', true},
            {"excel", 'e', true}
        };
        vector<tuple<string, char, bool>> listOptions = {
            {"report", 'r', true},
            {"list", 'l', false},
            {"quit", 'q', false}
        };
        auto parser = CommandParser();
        while (!finished) {
            if (indexList) {
                tie(option, index) = parser.parse(Colors::GREEN(), mainOptions, 'r', numFiles - 1);
            } else {
                tie(option, subIndex) = parser.parse(Colors::MAGENTA(), listOptions, 'r', results.at(index).load()["results"].size() - 1);
            }
            switch (option) {
                case 'q':
                    finished = true;
                    break;
                case 'l':
                    list();
                    indexList = true;
                    break;
                case 'd':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("delete", filename))
                        break;
                    cout << "Deleting " << filename << endl;
                    results.deleteResult(index);
                    cout << "File: " + filename + " deleted!" << endl;
                    list();
                    break;
                case 'h':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("hide", filename))
                        break;
                    filename = results.at(index).getFilename();
                    cout << "Hiding " << filename << endl;
                    results.hideResult(index, Paths::hiddenResults());
                    cout << "File: " + filename + " hidden! (moved to " << Paths::hiddenResults() << ")" << endl;
                    list();
                    break;
                case 's':
                    sortList();
                    list();
                    break;
                case 'r':
                    if (indexList) {
                        report(index, false);
                        indexList = false;
                    } else {
                        showIndex(index, subIndex);
                    }
                    break;
                case 'e':
                    report(index, true);
                    break;
            }
        }
    }
} /* namespace platform */
