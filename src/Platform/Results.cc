#include <filesystem>
#include "platformUtils.h"
#include "Results.h"
#include "Report.h"
#include "BestResult.h"
namespace platform {
    Result::Result(const string& path, const string& filename)
        : path(path)
        , filename(filename)
    {
        auto data = load();
        date = data["date"];
        score = 0;
        for (const auto& result : data["results"]) {
            score += result["score"].get<double>();
        }
        scoreName = data["score_name"];
        if (scoreName == BestResult::scoreName()) {
            score /= BestResult::score();
        }
        title = data["title"];
        duration = data["duration"];
        model = data["model"];
    }
    json Result::load()
    {
        ifstream resultData(path + "/" + filename);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw invalid_argument("Unable to open result file. [" + path + "/" + filename + "]");
    }
    void Results::load()
    {
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != string::npos && filename.find("results_") == 0) {
                auto result = Result(path, filename);
                bool addResult = true;
                if (model != "any" && result.getModel() != model || scoreName != "any" && scoreName != result.getScoreName())
                    addResult = false;
                if (addResult)
                    files.push_back(result);
            }
        }
    }
    string Result::to_string() const
    {
        stringstream oss;
        oss << date << " ";
        oss << setw(12) << left << model << " ";
        oss << setw(11) << left << scoreName << " ";
        oss << right << setw(11) << setprecision(7) << fixed << score << " ";
        oss << setw(9) << setprecision(3) << fixed << duration << " ";
        oss << setw(50) << left << title << " ";
        return  oss.str();
    }
    void Results::show() const
    {
        cout << "Results found: " << files.size() << endl;
        cout << "-------------------" << endl;
        auto i = 0;
        cout << " #  Date       Model        Score Name  Score       Duration  Title" << endl;
        cout << "=== ========== ============ =========== =========== ========= =============================================================" << endl;
        for (const auto& result : files) {
            cout << setw(3) << fixed << right << i++ << " ";
            cout << result.to_string() << endl;
            if (i == max && max != 0) {
                break;
            }

        }
    }
    int Results::getIndex(const string& intent) const
    {
        cout << "Choose result to " << intent << ": ";
        int index;
        cin >> index;
        if (index >= 0 && index < files.size()) {
            return index;
        }

        cout << "Invalid index" << endl;
        return -1;
    }
    void Results::menu()
    {
        cout << "Choose option (quit='q', list='l', delete='d', hide='h', sort='s', report='r'): ";
        char option;
        int index;
        string filename;
        cin >> option;
        switch (option) {
            case 'q':
                exit(0);
            case 'l':
                show();
                menu();
                break;
            case 'd':
                index = getIndex("delete");
                if (index == -1)
                    break;
                filename = files[index].getFilename();
                cout << "Deleting " << filename << endl;
                remove((path + "/" + filename).c_str());
                files.erase(files.begin() + index);
                show();
                menu();
                break;
            case 'h':
                index = getIndex("hide");
                if (index == -1)
                    break;
                filename = files[index].getFilename();
                cout << "Hiding " << filename << endl;
                rename((path + "/" + filename).c_str(), (path + "/." + filename).c_str());
                files.erase(files.begin() + index);
                show();
                menu();
                break;
            case 's':
                sortList();
                show();
                menu();
                break;
            case 'r':
                index = getIndex("report");
                if (index == -1)
                    break;
                filename = files[index].getFilename();
                cout << "Reporting " << filename << endl;
                auto data = files[index].load();
                Report report(data);
                report.show();
                menu();
                break;

        }
    }
    void Results::sortList()
    {
        cout << "Choose sorting field (date='d', score='s', duration='u', model='m'): ";
        char option;
        cin >> option;
        switch (option) {
            case 'd':
                sortDate();
                break;
            case 's':
                sortScore();
                break;
            case 'u':
                sortDuration();
                break;
            case 'm':
                sortModel();
                break;
            default:
                cout << "Invalid option" << endl;
        }

    }
    void Results::sortDate()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDate() > b.getDate();
            });
    }
    void Results::sortModel()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getModel() > b.getModel();
            });
    }
    void Results::sortDuration()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDuration() > b.getDuration();
            });
    }
    void Results::sortScore()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getScore() > b.getScore();
            });
    }
    void Results::manage()
    {
        if (files.size() == 0) {
            cout << "No results found!" << endl;
            exit(0);
        }
        show();
        menu();
    }

}