#include <sstream>
#include <locale>
#include "ReportConsole.h"
#include "BestScore.h"


namespace platform {
    struct separated : numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        string do_grouping() const { return "\03"; }
    };

    string ReportConsole::headerLine(const string& text, int utf = 0)
    {
        int n = MAXL - text.length() - 3;
        n = n < 0 ? 0 : n;
        return "* " + text + string(n + utf, ' ') + "*\n";
    }

    void ReportConsole::header()
    {
        locale mylocale(cout.getloc(), new separated);
        locale::global(mylocale);
        cout.imbue(mylocale);
        stringstream oss;
        cout << Colors::MAGENTA() << string(MAXL, '*') << endl;
        cout << headerLine("Report " + data["model"].get<string>() + " ver. " + data["version"].get<string>() + " with " + to_string(data["folds"].get<int>()) + " Folds cross validation and " + to_string(data["seeds"].size()) + " random seeds. " + data["date"].get<string>() + " " + data["time"].get<string>());
        cout << headerLine(data["title"].get<string>());
        cout << headerLine("Random seeds: " + fromVector("seeds") + " Stratified: " + (data["stratified"].get<bool>() ? "True" : "False"));
        oss << "Execution took  " << setprecision(2) << fixed << data["duration"].get<float>() << " seconds,   " << data["duration"].get<float>() / 3600 << " hours, on " << data["platform"].get<string>();
        cout << headerLine(oss.str());
        cout << headerLine("Score is " + data["score_name"].get<string>());
        cout << string(MAXL, '*') << endl;
        cout << endl;
    }
    void ReportConsole::body()
    {
        cout << Colors::GREEN() << " #  Dataset                   Sampl. Feat. Cls Nodes     Edges     States    Score           Time                Hyperparameters" << endl;
        cout << "=== ========================= ====== ===== === ========= ========= ========= =============== =================== ====================" << endl;
        json lastResult;
        double totalScore = 0.0;
        bool odd = true;
        int index = 0;
        for (const auto& r : data["results"]) {
            if (selectedIndex != -1 && index != selectedIndex) {
                index++;
                continue;
            }
            auto color = odd ? Colors::CYAN() : Colors::BLUE();
            cout << color;
            cout << setw(3) << index++ << " ";
            cout << setw(25) << left << r["dataset"].get<string>() << " ";
            cout << setw(6) << right << r["samples"].get<int>() << " ";
            cout << setw(5) << right << r["features"].get<int>() << " ";
            cout << setw(3) << right << r["classes"].get<int>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["nodes"].get<float>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["leaves"].get<float>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["depth"].get<float>() << " ";
            cout << setw(8) << right << setprecision(6) << fixed << r["score"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["score_std"].get<double>();
            const string status = compareResult(r["dataset"].get<string>(), r["score"].get<double>());
            cout << status;
            cout << setw(12) << right << setprecision(6) << fixed << r["time"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["time_std"].get<double>() << " ";
            try {
                cout << r["hyperparameters"].get<string>();
            }
            catch (const exception& err) {
                cout << r["hyperparameters"];
            }
            cout << endl;
            lastResult = r;
            totalScore += r["score"].get<double>();
            odd = !odd;
        }
        if (data["results"].size() == 1 || selectedIndex != -1) {
            cout << string(MAXL, '*') << endl;
            cout << headerLine(fVector("Train scores: ", lastResult["scores_train"], 14, 12));
            cout << headerLine(fVector("Test  scores: ", lastResult["scores_test"], 14, 12));
            cout << headerLine(fVector("Train  times: ", lastResult["times_train"], 10, 3));
            cout << headerLine(fVector("Test   times: ", lastResult["times_test"], 10, 3));
            cout << string(MAXL, '*') << endl;
        } else {
            footer(totalScore);
        }
    }
    void ReportConsole::showSummary()
    {
        for (const auto& item : summary) {
            stringstream oss;
            oss << setw(3) << left << item.first;
            oss << setw(3) << right << item.second << " ";
            oss << left << meaning.at(item.first);
            cout << headerLine(oss.str(), 2);
        }
    }

    void ReportConsole::footer(double totalScore)
    {
        cout << Colors::MAGENTA() << string(MAXL, '*') << endl;
        showSummary();
        auto score = data["score_name"].get<string>();
        if (score == BestScore::scoreName()) {
            stringstream oss;
            oss << score << " compared to " << BestScore::title() << " .:  " << totalScore / BestScore::score();
            cout << headerLine(oss.str());
        }
        if (!getExistBestFile() && compare) {
            cout << headerLine("*** Best Results File not found. Couldn't compare any result!");
        }
        cout << string(MAXL, '*') << endl << Colors::RESET();
    }
}