#include <filesystem>
#include <fstream>
#include <iostream>
#include "BestResults.h"
#include "Result.h"
#include "Colors.h"

namespace platform {

    void BestResults::build()
    {
        auto files = loadFiles();
        if (files.size() == 0) {
            throw runtime_error("No result files were found!");
        }
        json bests;
        for (const auto& file : files) {
            auto result = Result(path, file);
            auto data = result.load();
            for (auto const& item : data.at("results")) {
                bool update = false;
                if (bests.contains(item.at("dataset").get<string>())) {
                    if (item.at("score").get<double>() > bests["dataset"].at(0).get<double>()) {
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
        string bestFileName = path + "/" + bestResultFile();
        if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
            fclose(fileTest);
            cout << Colors::MAGENTA() << "File " << bestFileName << " already exists and it shall be overwritten." << Colors::RESET();
        }
        ofstream file(bestFileName);
        file << bests;
        file.close();
    }

    string BestResults::bestResultFile()
    {
        return "best_results_" + score + "_" + model + ".json";
    }

    vector<string> BestResults::loadFiles()
    {
        vector<string> files;
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto fileName = file.path().filename().string();
            if (fileName.find(".json") != string::npos && fileName.find("results_") == 0
                && fileName.find("_" + score + "_") != string::npos
                && fileName.find("_" + model + "_") != string::npos) {
                files.push_back(fileName);
            }
        }
        return files;
    }

    void BestResults::report()
    {

    }
}