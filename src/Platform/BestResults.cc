#include <filesystem>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include "BestResults.h"
#include "Result.h"
#include "Colors.h"
#include "Statistics.h"
#include "BestResultsExcel.h"


namespace fs = std::filesystem;
// function ftime_to_string, Code taken from 
// https://stackoverflow.com/a/58237530/1389271
template <typename TP>
std::string ftime_to_string(TP tp)
{
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now()
        + system_clock::now());
    auto tt = system_clock::to_time_t(sctp);
    std::tm* gmt = std::gmtime(&tt);
    std::stringstream buffer;
    buffer << std::put_time(gmt, "%Y-%m-%d %H:%M");
    return buffer.str();
}
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
                // Check if results file contains only one dataset
                auto datasetName = item.at("dataset").get<string>();
                if (bests.contains(datasetName)) {
                    if (item.at("score").get<double>() > bests[datasetName].at(0).get<double>()) {
                        update = true;
                    }
                } else {
                    update = true;
                }
                if (update) {
                    bests[datasetName] = { item.at("score").get<double>(), item.at("hyperparameters"), file };
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
        return "best_results_" + score + "_" + model + ".json";
    }

    pair<string, string> getModelScore(string name)
    {
        // results_accuracy_BoostAODE_MacBookpro16_2023-09-06_12:27:00_1.json
        int i = 0;
        auto pos = name.find("_");
        auto pos2 = name.find("_", pos + 1);
        string score = name.substr(pos + 1, pos2 - pos - 1);
        pos = name.find("_", pos2 + 1);
        string model = name.substr(pos2 + 1, pos - pos2 - 1);
        return { model, score };
    }

    vector<string> BestResults::loadResultFiles()
    {
        vector<string> files;
        using std::filesystem::directory_iterator;
        string fileModel, fileScore;
        for (const auto& file : directory_iterator(path)) {
            auto fileName = file.path().filename().string();
            if (fileName.find(".json") != string::npos && fileName.find("results_") == 0) {
                tie(fileModel, fileScore) = getModelScore(fileName);
                if (score == fileScore && (model == fileModel || model == "any")) {
                    files.push_back(fileName);
                }
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
    vector<string> BestResults::getModels()
    {
        set<string> models;
        vector<string> result;
        auto files = loadResultFiles();
        if (files.size() == 0) {
            cerr << Colors::MAGENTA() << "No result files were found!" << Colors::RESET() << endl;
            exit(1);
        }
        string fileModel, fileScore;
        for (const auto& file : files) {
            // extract the model from the file name
            tie(fileModel, fileScore) = getModelScore(file);
            // add the model to the vector of models
            models.insert(fileModel);
        }
        result = vector<string>(models.begin(), models.end());
        return result;
    }
    vector<string> BestResults::getDatasets(json table)
    {
        vector<string> datasets;
        for (const auto& dataset : table.items()) {
            datasets.push_back(dataset.key());
        }
        return datasets;
    }

    void BestResults::buildAll()
    {
        auto models = getModels();
        for (const auto& model : models) {
            cout << "Building best results for model: " << model << endl;
            this->model = model;
            build();
        }
        model = "any";
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
        auto date = ftime_to_string(filesystem::last_write_time(bestFileName));
        auto data = loadFile(bestFileName);
        auto datasets = getDatasets(data);
        int maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        cout << Colors::GREEN() << "Best results for " << model << " and " << score << " as of " << date << endl;
        cout << "--------------------------------------------------------" << endl;
        cout << Colors::GREEN() << " #  " << setw(maxDatasetName + 1) << left << string("Dataset") << "Score       File                                                               Hyperparameters" << endl;
        cout << "=== " << string(maxDatasetName, '=') << " =========== ================================================================== ================================================= " << endl;
        auto i = 0;
        bool odd = true;
        for (auto const& item : data.items()) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << setw(maxDatasetName) << left << item.key() << " ";
            cout << setw(11) << setprecision(9) << fixed << item.value().at(0).get<double>() << " ";
            cout << setw(66) << item.value().at(2).get<string>() << " ";
            cout << item.value().at(1) << " ";
            cout << endl;
            odd = !odd;
        }
    }
    json BestResults::buildTableResults(vector<string> models)
    {
        json table;
        auto maxDate = filesystem::file_time_type::max();
        for (const auto& model : models) {
            this->model = model;
            string bestFileName = path + bestResultFile();
            if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
                fclose(fileTest);
            } else {
                cerr << Colors::MAGENTA() << "File " << bestFileName << " doesn't exist." << Colors::RESET() << endl;
                exit(1);
            }
            auto dateWrite = filesystem::last_write_time(bestFileName);
            if (dateWrite < maxDate) {
                maxDate = dateWrite;
            }
            auto data = loadFile(bestFileName);
            table[model] = data;
        }
        table["dateTable"] = ftime_to_string(maxDate);
        return table;
    }

    void BestResults::printTableResults(vector<string> models, json table)
    {
        cout << Colors::GREEN() << "Best results for " << score << " as of " << table.at("dateTable").get<string>() << endl;
        cout << "------------------------------------------------" << endl;
        cout << Colors::GREEN() << " #  " << setw(maxDatasetName + 1) << left << string("Dataset");
        for (const auto& model : models) {
            cout << setw(maxModelName) << left << model << " ";
        }
        cout << endl;
        cout << "=== " << string(maxDatasetName, '=') << " ";
        for (const auto& model : models) {
            cout << string(maxModelName, '=') << " ";
        }
        cout << endl;
        auto i = 0;
        bool odd = true;
        map<string, double> totals;
        int nDatasets = table.begin().value().size();
        for (const auto& model : models) {
            totals[model] = 0.0;
        }
        auto datasets = getDatasets(table.begin().value());
        for (auto const& dataset : datasets) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << setw(maxDatasetName) << left << dataset << " ";
            double maxValue = 0;
            // Find out the max value for this dataset
            for (const auto& model : models) {
                double value = table[model].at(dataset).at(0).get<double>();
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            // Print the row with red colors on max values
            for (const auto& model : models) {
                string efectiveColor = color;
                double value = table[model].at(dataset).at(0).get<double>();
                if (value == maxValue) {
                    efectiveColor = Colors::RED();
                }
                totals[model] += value;
                cout << efectiveColor << setw(maxModelName) << setprecision(maxModelName - 2) << fixed << value << " ";
            }
            cout << endl;
            odd = !odd;
        }
        cout << Colors::GREEN() << "=== " << string(maxDatasetName, '=') << " ";
        for (const auto& model : models) {
            cout << string(maxModelName, '=') << " ";
        }
        cout << endl;
        cout << Colors::GREEN() << setw(5 + maxDatasetName) << "    Totals...................";
        double max = 0.0;
        for (const auto& total : totals) {
            if (total.second > max) {
                max = total.second;
            }
        }
        for (const auto& model : models) {
            string efectiveColor = Colors::GREEN();
            if (totals[model] == max) {
                efectiveColor = Colors::RED();
            }
            cout << efectiveColor << right << setw(maxModelName) << setprecision(maxModelName - 4) << fixed << totals[model] << " ";
        }
        cout << endl;
    }
    void BestResults::reportAll(bool excel)
    {
        auto models = getModels();
        // Build the table of results
        json table = buildTableResults(models);
        vector<string> datasets = getDatasets(table.begin().value());
        maxModelName = (*max_element(models.begin(), models.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        maxModelName = max(12, maxModelName);
        maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        maxDatasetName = max(25, maxDatasetName);
        // Print the table of results
        printTableResults(models, table);
        // Compute the Friedman test
        if (friedman) {
            Statistics stats(models, datasets, table, significance);
            auto result = stats.friedmanTest();
            stats.postHocHolmTest(result);
        }
        if (excel) {
            BestResultsExcel excel(score, models, datasets, table, friedman, significance);
            excel.build();
            cout << Colors::YELLOW() << "** Excel file generated: " << excel.getFileName() << Colors::RESET() << endl;
        }
    }
}