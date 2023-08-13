#include <filesystem>
#include "platformUtils.h"
#include "Results.h"
namespace platform {
    const double REFERENCE_SCORE = 22.109799;
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
        score /= REFERENCE_SCORE;
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
                files.push_back(result);
            }
        }
    }
    string Result::to_string() const
    {
        stringstream oss;
        oss << date << " ";
        oss << setw(12) << left << model << " ";
        oss << right << setw(9) << setprecision(7) << fixed << score << " ";
        oss << setw(9) << setprecision(3) << fixed << duration << " ";
        oss << setw(50) << left << title << " ";
        return  oss.str();
    }
    void Results::manage()
    {
        cout << "Results found: " << files.size() << endl;
        cout << "========================" << endl;
        for (const auto& result : files) {
            cout << result.to_string() << endl;
        }
    }

}