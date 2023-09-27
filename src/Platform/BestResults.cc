#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include "BestResults.h"
#include "Result.h"
#include "Colors.h"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>



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
struct WTL {
    int win;
    int tie;
    int loss;
};

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
        auto date = ftime_to_string(filesystem::last_write_time(bestFileName));
        auto data = loadFile(bestFileName);
        cout << Colors::GREEN() << "Best results for " << model << " and " << score << " as of " << date << endl;
        cout << "--------------------------------------------------------" << endl;
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
        table["dateTable"] = ftime_to_string(maxDate);
        return table;
    }
    map<string, float> assignRanks(vector<pair<string, double>>& ranksOrder)
    {
        // sort the ranksOrder vector by value
        sort(ranksOrder.begin(), ranksOrder.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
            return a.second > b.second;
            });
        //Assign ranks to  values and if they are the same they share the same averaged rank
        map<string, float> ranks;
        for (int i = 0; i < ranksOrder.size(); i++) {
            ranks[ranksOrder[i].first] = i + 1.0;
        }
        int i = 0;
        while (i < static_cast<int>(ranksOrder.size())) {
            int j = i + 1;
            int sumRanks = ranks[ranksOrder[i].first];
            while (j < static_cast<int>(ranksOrder.size()) && ranksOrder[i].second == ranksOrder[j].second) {
                sumRanks += ranks[ranksOrder[j++].first];
            }
            if (j > i + 1) {
                float averageRank = (float)sumRanks / (j - i);
                for (int k = i; k < j; k++) {
                    ranks[ranksOrder[k].first] = averageRank;
                }
            }
            i = j;
        }
        return ranks;
    }

    map<int, WTL> computeWTL(int controlIdx, vector<string> models, json table)
    {
        // Compute the WTL matrix
        map<int, WTL> wtl;
        int nModels = models.size();
        for (int i = 0; i < nModels; ++i) {
            wtl[i] = { 0, 0, 0 };
        }
        json origin = table.begin().value();
        for (auto const& item : origin.items()) {
            auto controlModel = models.at(controlIdx);
            double controlValue = table[controlModel].at(item.key()).at(0).get<double>();
            for (int i = 0; i < nModels; ++i) {
                if (i == controlIdx) {
                    continue;
                }
                double value = table[models[i]].at(item.key()).at(0).get<double>();
                if (value < controlValue) {
                    wtl[i].win++;
                } else if (value == controlValue) {
                    wtl[i].tie++;
                } else {
                    wtl[i].loss++;
                }
            }
        }
        return wtl;
    }

    void postHocHolm(int controlIdx, vector<string> models, int nDatasets, map<string, float> ranks, double significance, map<int, WTL> wtl)
    {
        // Reference https://link.springer.com/article/10.1007/s44196-022-00083-8
        // Post-hoc Holm test
        // Calculate the p-value for the models paired with the control model
        int nModels = models.size();
        map<int, double> stats; // p-value of each model paired with the control model
        boost::math::normal dist(0.0, 1.0);
        double diff = sqrt(nModels * (nModels + 1) / (6.0 * nDatasets));
        for (int i = 0; i < nModels; i++) {
            if (i == controlIdx) {
                stats[i] = 0.0;
                continue;
            }
            double z = abs(ranks.at(models[controlIdx]) - ranks.at(models[i])) / diff;
            double p_value = (long double)2 * (1 - cdf(dist, z));
            stats[i] = p_value;
        }
        // Sort the models by p-value
        vector<pair<int, double>> statsOrder;
        for (const auto& stat : stats) {
            statsOrder.push_back({ stat.first, stat.second });
        }
        sort(statsOrder.begin(), statsOrder.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
            return a.second < b.second;
            });

        // Holm adjustment
        for (int i = 0; i < statsOrder.size(); ++i) {
            auto item = statsOrder.at(i);
            double before = i == 0 ? 0.0 : statsOrder.at(i - 1).second;
            double p_value = min((double)1.0, item.second * (nModels - i));
            p_value = max(before, p_value);
            statsOrder[i] = { item.first, p_value };
        }
        cout << Colors::CYAN();
        cout << "  *************************************************************************************************************" << endl;
        cout << "  Post-hoc Holm test: H0: 'There is no significant differences between the control model and the other models.'" << endl;
        cout << "  Control model: " << models[controlIdx] << endl;
        cout << "  Model        p-value      rank      win tie loss" << endl;
        cout << "  ============ ============ ========= === === ====" << endl;
        // sort ranks from lowest to highest
        vector<pair<string, float>> ranksOrder;
        for (const auto& rank : ranks) {
            ranksOrder.push_back({ rank.first, rank.second });
        }
        sort(ranksOrder.begin(), ranksOrder.end(), [](const pair<string, float>& a, const pair<string, float>& b) {
            return a.second < b.second;
            });
        for (const auto& item : ranksOrder) {
            if (item.first == models.at(controlIdx)) {
                continue;
            }
            auto idx = distance(models.begin(), find(models.begin(), models.end(), item.first));
            double pvalue = 0.0;
            for (const auto& stat : statsOrder) {
                if (stat.first == idx) {
                    pvalue = stat.second;
                }
            }
            cout << "  " << left << setw(12) << item.first << " " << setprecision(10) << fixed << pvalue << setprecision(7) << " " << item.second;
            cout << " " << right << setw(3) << wtl.at(idx).win << " " << setw(3) << wtl.at(idx).tie << " " << setw(4) << wtl.at(idx).loss << endl;
        }
        cout << "  *************************************************************************************************************" << endl;
        cout << Colors::RESET();
    }
    bool friedmanTest(vector<string> models, int nDatasets, map<string, float> ranks, double significance = 0.05)
    {
        // Friedman test
        // Calculate the Friedman statistic
        int nModels = models.size();
        if (nModels < 3 || nDatasets < 3) {
            throw runtime_error("Can't make the Friedman test with less than 3 models and/or less than 3 datasets.");
        }
        cout << Colors::BLUE() << endl;
        cout << "***************************************************************************************************************" << endl;
        cout << Colors::GREEN() << "Friedman test: H0: 'There is no significant differences between all the classifiers.'" << Colors::BLUE() << endl;
        double degreesOfFreedom = nModels - 1.0;
        double sumSquared = 0;
        for (const auto& rank : ranks) {
            sumSquared += pow(rank.second, 2);
        }
        // Compute the Friedman statistic as in https://link.springer.com/article/10.1007/s44196-022-00083-8
        double friedmanQ = 12.0 * nDatasets / (nModels * (nModels + 1)) * (sumSquared - (nModels * pow(nModels + 1, 2)) / 4);
        cout << "Friedman statistic: " << friedmanQ << endl;
        // Calculate the critical value
        boost::math::chi_squared chiSquared(degreesOfFreedom);
        long double p_value = (long double)1.0 - cdf(chiSquared, friedmanQ);
        double criticalValue = quantile(chiSquared, 1 - significance);
        std::cout << "Critical Chi-Square Value for df=" << fixed << (int)degreesOfFreedom
            << " and alpha=" << setprecision(2) << fixed << significance << ": " << setprecision(7) << scientific << criticalValue << std::endl;
        cout << "p-value: " << scientific << p_value << " is " << (p_value < significance ? "less" : "greater") << " than " << setprecision(2) << fixed << significance << endl;
        bool result;
        if (p_value < significance) {
            cout << Colors::GREEN() << "The null hypothesis H0 is rejected." << endl;
            result = true;
        } else {
            cout << Colors::YELLOW() << "The null hypothesis H0 is accepted. Computed p-values will not be significant." << endl;
            result = false;
        }
        cout << Colors::BLUE() << "***************************************************************************************************************" << endl;
        return result;
    }
    void BestResults::printTableResults(set<string> models, json table)
    {
        cout << Colors::GREEN() << "Best results for " << score << " as of " << table.at("dateTable").get<string>() << endl;
        cout << "------------------------------------------------" << endl;
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
        map<string, float> ranks;
        map<string, float> ranksTotal;
        int nDatasets = table.begin().value().size();
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
            // Assign the ranks
            ranks = assignRanks(ranksOrder);
            if (ranksTotal.size() == 0) {
                ranksTotal = ranks;
            } else {
                for (const auto& rank : ranks) {
                    ranksTotal[rank.first] += rank.second;
                }
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
                // cout << efectiveColor << setw(12) << setprecision(10) << fixed << ranks[model] << " ";
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
            cout << efectiveColor << setw(12) << setprecision(9) << fixed << totals[model] << " ";
        }
        // Output the averaged ranks
        cout << endl;
        int min = 1;
        for (auto& rank : ranksTotal) {
            if (rank.second < min) {
                min = rank.second;
            }
            rank.second /= nDatasets;
        }
        cout << Colors::BLUE() << setw(30) << "    Ranks....................";
        for (const auto& model : models) {
            string efectiveColor = Colors::BLUE();
            if (ranksTotal[model] == min) {
                efectiveColor = Colors::RED();
            }
            cout << efectiveColor << setw(12) << setprecision(4) << fixed << (double)ranksTotal[model] << " ";
        }
        cout << endl;
        cout << Colors::GREEN() << setw(30) << "    Averaged ranks...........";
        for (const auto& model : models) {
            string efectiveColor = Colors::GREEN();
            if (ranksTotal[model] == min) {
                efectiveColor = Colors::RED();
            }
            cout << efectiveColor << setw(12) << setprecision(9) << fixed << (double)ranksTotal[model] << " ";
        }
        cout << endl;
        if (friedman) {
            double significance = 0.05;
            vector<string> vModels(models.begin(), models.end());
            friedmanTest(vModels, nDatasets, ranksTotal, significance);
            // Stablish the control model as the one with the lowest averaged rank
            int controlIdx = distance(ranks.begin(), min_element(ranks.begin(), ranks.end(), [](const auto& l, const auto& r) { return l.second < r.second; }));
            auto wtl = computeWTL(controlIdx, vModels, table);
            postHocHolm(controlIdx, vModels, nDatasets, ranksTotal, significance, wtl);
        }
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