#include <sstream>
#include <locale>
#include "Report.h"
#include "BestResult.h"


namespace platform {
    string headerLine(const string& text)
    {
        int n = MAXL - text.length() - 3;
        n = n < 0 ? 0 : n;
        return "* " + text + string(n, ' ') + "*\n";
    }
    string Report::fromVector(const string& key)
    {
        string result = "";

        for (auto& item : data[key]) {
            result += to_string(item) + ", ";
        }
        return "[" + result.substr(0, result.size() - 2) + "]";
    }
    string fVector(const json& data)
    {
        string result = "";
        for (const auto& item : data) {
            result += to_string(item) + ", ";
        }
        return "[" + result.substr(0, result.size() - 2) + "]";
    }
    void Report::show()
    {
        header();
        body();
        footer();
    }
    struct separated : numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        string do_grouping() const { return "\03"; }
    };
    void Report::header()
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
    void Report::body()
    {
        cout << Colors::GREEN() << "Dataset                        Sampl. Feat. Cls Nodes     Edges     States    Score           Time               Hyperparameters" << endl;
        cout << "============================== ====== ===== === ========= ========= ========= =============== ================== ===============" << endl;
        json lastResult;
        totalScore = 0;
        bool odd = true;
        for (const auto& r : data["results"]) {
            auto color = odd ? Colors::CYAN() : Colors::BLUE();
            cout << color << setw(30) << left << r["dataset"].get<string>() << " ";
            cout << setw(6) << right << r["samples"].get<int>() << " ";
            cout << setw(5) << right << r["features"].get<int>() << " ";
            cout << setw(3) << right << r["classes"].get<int>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["nodes"].get<float>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["leaves"].get<float>() << " ";
            cout << setw(9) << setprecision(2) << fixed << r["depth"].get<float>() << " ";
            cout << setw(8) << right << setprecision(6) << fixed << r["score"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["score_std"].get<double>() << " ";
            cout << setw(11) << right << setprecision(6) << fixed << r["time"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["time_std"].get<double>() << " ";
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
        if (data["results"].size() == 1) {
            cout << string(MAXL, '*') << endl;
            cout << headerLine("Train scores: " + fVector(lastResult["scores_train"]));
            cout << headerLine("Test  scores: " + fVector(lastResult["scores_test"]));
            cout << headerLine("Train  times: " + fVector(lastResult["times_train"]));
            cout << headerLine("Test   times: " + fVector(lastResult["times_test"]));
            cout << string(MAXL, '*') << endl;
        }
    }
    void Report::footer()
    {
        cout << Colors::MAGENTA() << string(MAXL, '*') << endl;
        auto score = data["score_name"].get<string>();
        if (score == BestResult::scoreName()) {
            stringstream oss;
            oss << score << " compared to " << BestResult::title() << " .:  " << totalScore / BestResult::score();
            cout << headerLine(oss.str());
        }
        cout << string(MAXL, '*') << endl << Colors::RESET();

    }
}