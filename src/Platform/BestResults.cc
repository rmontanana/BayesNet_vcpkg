#include <filesystem>
#include <fstream>
#include <iostream>
#include "BestResults.h"
#include "Result.h"
#include "Colors.h"

namespace platform {

    string BestResults::build()
    {
        auto files = loadResultFiles();
        if (files.size() == 0) {
            cerr << Colors::MAGENTA() << "No result files were found!" << Colors::RESET() << endl;
            exit(1);
        }
        json bests;
        for (const auto& file : files) {
            auto result = Result(path, file);
            auto data = result.load();
            for (auto const& item : data.at("results")) {
                bool update = false;
                if (bests.contains(item.at("dataset").get<string>())) {
                    if (item.at("score").get<double>() > bests[item.at("dataset").get<string>()].at(0).get<double>()) {
                        update = true;
                    }
                } else {
                    update = true;
                }
                if (update) {
                    bests[item.at("dataset").get<string>()] = { item.at("score").get<double>(), item.at("hyperparameters"), file };
                }
            }
        }
        string bestFileName = path + bestResultFile();
        if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
            fclose(fileTest);
            cout << Colors::MAGENTA() << "File " << bestFileName << " already exists and it shall be overwritten." << Colors::RESET() << endl;
        }
        ofstream file(bestFileName);
        file << bests;
        file.close();
        return bestFileName;
    }

    string BestResults::bestResultFile()
    {
        return "best_results_" + model + "_" + score + ".json";
    }

    vector<string> BestResults::loadResultFiles()
    {
        vector<string> files;
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto fileName = file.path().filename().string();
            if (fileName.find(".json") != string::npos && fileName.find("results_") == 0
                && fileName.find("_" + score + "_") != string::npos
                && (fileName.find("_" + model + "_") != string::npos || model == "any")) {
                files.push_back(fileName);
            }
        }
        return files;
    }
    json BestResults::loadFile(const string& fileName)
    {
        ifstream resultData(fileName);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw invalid_argument("Unable to open result file. [" + fileName + "]");
    }
    void BestResults::reportAll()
    {

    }

    void BestResults::reportSingle()
    {
        string bestFileName = path + bestResultFile();
        if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
            fclose(fileTest);
        } else {
            cerr << Colors::MAGENTA() << "File " << bestFileName << " doesn't exist." << Colors::RESET() << endl;
            exit(1);
        }
        auto data = loadFile(bestFileName);
        cout << Colors::GREEN() << "Best results for " << model << " and " << score << endl;
        cout << "------------------------------------------" << endl;
        cout << Colors::GREEN() << " #  Dataset                   Score       File                                                               Hyperparameters" << endl;
        cout << "=== ========================= =========== ================================================================== ================================================= " << endl;
        auto i = 0;
        bool odd = true;
        for (auto const& item : data.items()) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << setw(25) << left << item.key() << " ";
            cout << setw(11) << setprecision(9) << fixed << item.value().at(0).get<double>() << " ";
            cout << setw(66) << item.value().at(2).get<string>() << " ";
            cout << item.value().at(1) << " ";
            cout << endl;
            odd = !odd;
        }


    }
}