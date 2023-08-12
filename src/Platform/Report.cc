#include "Report.h"

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
    }
    void Report::header()
    {
        cout << string(MAXL, '*') << endl;
        cout << headerLine("Report " + data["model"].get<string>() + " ver. " + data["version"].get<string>() + " with " + to_string(data["folds"].get<int>()) + " Folds cross validation and " + to_string(data["seeds"].size()) + " random seeds. " + data["date"].get<string>() + " " + data["time"].get<string>());
        cout << headerLine(data["title"].get<string>());
        cout << headerLine("Random seeds: " + fromVector("seeds") + " Stratified: " + (data["stratified"].get<bool>() ? "True" : "False"));
        cout << headerLine("Execution took  " + to_string(data["duration"].get<float>()) + " seconds,   " + to_string(data["duration"].get<float>() / 3600) + " hours, on " + data["platform"].get<string>());
        cout << headerLine("Score is " + data["score_name"].get<string>());
        cout << string(MAXL, '*') << endl;
        cout << endl;
    }
    void Report::body()
    {
        cout << "Dataset                        Sampl. Feat. Cls Nodes   Edges   States  Score           Time              Hyperparameters" << endl;
        cout << "============================== ====== ===== === ======= ======= ======= =============== ================= ===============" << endl;
        for (const auto& r : data["results"]) {
            cout << setw(30) << left << r["dataset"].get<string>() << " ";
            cout << setw(6) << right << r["samples"].get<int>() << " ";
            cout << setw(5) << right << r["features"].get<int>() << " ";
            cout << setw(3) << right << r["classes"].get<int>() << " ";
            cout << setw(7) << setprecision(2) << fixed << r["nodes"].get<float>() << " ";
            cout << setw(7) << setprecision(2) << fixed << r["leaves"].get<float>() << " ";
            cout << setw(7) << setprecision(2) << fixed << r["depth"].get<float>() << " ";
            cout << setw(8) << right << setprecision(6) << fixed << r["score_test"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["score_test_std"].get<double>() << " ";
            cout << setw(10) << right << setprecision(6) << fixed << r["test_time"].get<double>() << "±" << setw(6) << setprecision(4) << fixed << r["test_time_std"].get<double>() << " ";
            cout << " " << r["hyperparameters"].get<string>();
            cout << endl;
            cout << string(MAXL, '*') << endl;
            cout << headerLine("Train scores: " + fVector(r["scores_train"]));
            cout << headerLine("Test  scores: " + fVector(r["scores_test"]));
            cout << headerLine("Train  times: " + fVector(r["times_train"]));
            cout << headerLine("Test   times: " + fVector(r["times_test"]));
            cout << string(MAXL, '*') << endl;
        }
    }
}