#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
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
    set<string> BestResults::getModels()
    {
        set<string> models;
        auto files = loadResultFiles();
        if (files.size() == 0) {
            cerr << Colors::MAGENTA() << "No result files were found!" << Colors::RESET() << endl;
            exit(1);
        }
        string fileModel, fileScore;
        for (const auto& file : files) {
            // take the model from the file name and add it to a vector of models 
            // set model to the name of the first model in the vector
            // filter files and build the best results file of this model
            // repeat for all models
            // another for loop to read the best results file of each model and print al together
            // each row is a dataset and each column is a model
            // the score is the score of the best result of each model for that dataset
            // the rows are datasets the columns are models and the cells are the scores
            // the first row is the header with the model names
            // the first column is the dataset names
            // the last column is the average score of each dataset
            // the last row is the average score of each model
            // the last cell is the average score of all models
            // the last row and column are in bold

            // extract the model from the file name
            tie(fileModel, fileScore) = getModelScore(file);
            // add the model to the vector of models
            models.insert(fileModel);
        }
        return models;
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
    json BestResults::buildTableResults(set<string> models)
    {
        int numberOfDatasets = 0;
        bool first = true;
        json origin;
        json table;
        for (const auto& model : models) {
            this->model = model;
            string bestFileName = path + bestResultFile();
            if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
                fclose(fileTest);
            } else {
                cerr << Colors::MAGENTA() << "File " << bestFileName << " doesn't exist." << Colors::RESET() << endl;
                exit(1);
            }
            auto data = loadFile(bestFileName);
            if (first) {
                // Get the number of datasets of the first file and check that is the same for all the models
                first = false;
                numberOfDatasets = data.size();
                origin = data;
            } else {
                if (numberOfDatasets != data.size()) {
                    cerr << Colors::MAGENTA() << "The number of datasets in the best results files is not the same for all the models." << Colors::RESET() << endl;
                    exit(1);
                }
            }
            table[model] = data;
        }
        return table;
    }
    void BestResults::printTableResults(set<string> models, json table)
    {
        cout << Colors::GREEN() << "Best results for " << score << endl;
        cout << "------------------------------------------" << endl;
        cout << Colors::GREEN() << " #  Dataset                   ";
        for (const auto& model : models) {
            cout << setw(12) << left << model << " ";
        }
        cout << endl;
        cout << "=== ========================= ";
        for (const auto& model : models) {
            cout << "============ ";
        }
        cout << endl;
        auto i = 0;
        bool odd = true;
        map<string, double> totals;
        map<string, int> ranks;
        for (const auto& model : models) {
            totals[model] = 0.0;
        }
        json origin = table.begin().value();
        for (auto const& item : origin.items()) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            cout << color << setw(3) << fixed << right << i++ << " ";
            cout << setw(25) << left << item.key() << " ";
            double maxValue = 0;
            vector<pair<string, double>> ranksOrder;
            // Find out the max value for this dataset
            for (const auto& model : models) {
                double value = table[model].at(item.key()).at(0).get<double>();
                if (value > maxValue) {
                    maxValue = value;
                }
                ranksOrder.push_back({ model, value });
            }
            // sort the ranksOrder vector by value
            sort(ranksOrder.begin(), ranksOrder.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second > b.second;
                });
            // Assign the ranks
            for (int i = 0; i < ranksOrder.size(); i++) {
                ranks[ranksOrder[i].first] = i + 1;
            }
            // Print the row with red colors on max values
            for (const auto& model : models) {
                string efectiveColor = color;
                double value = table[model].at(item.key()).at(0).get<double>();
                if (value == maxValue) {
                    efectiveColor = Colors::RED();
                }
                totals[model] += value;
                cout << efectiveColor << setw(12) << setprecision(10) << fixed << value << " ";
            }
            cout << endl;
            odd = !odd;
        }
        cout << Colors::GREEN() << "=== ========================= ";
        for (const auto& model : models) {
            cout << "============ ";
        }
        cout << endl;
        cout << Colors::GREEN() << setw(30) << "    Totals...................";
        for (const auto& model : models) {
            cout << setw(12) << setprecision(9) << fixed << totals[model] << " ";
        }
        // Output the averaged ranks
        cout << endl;
        cout << Colors::GREEN() << setw(30) << "    Averaged ranks...........";
        for (const auto& model : models) {
            cout << setw(12) << setprecision(10) << fixed << (double)ranks[model] / (double)origin.size() << " ";
        }
        cout << endl;
    }
    void BestResults::reportAll()
    {
        auto models = getModels();
        // Build the table of results
        json table = buildTableResults(models);
        // Print the table of results
        printTableResults(models, table);
    }
}