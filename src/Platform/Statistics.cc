#include <sstream>
#include "Statistics.h"
#include "Colors.h"
#include "Symbols.h"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

namespace platform {

    Statistics::Statistics(const vector<string>& models, const vector<string>& datasets, const json& data, double significance, bool output) :
        models(models), datasets(datasets), data(data), significance(significance), output(output)
    {
        nModels = models.size();
        nDatasets = datasets.size();
    };

    void Statistics::fit()
    {
        if (nModels < 3 || nDatasets < 3) {
            cerr << "nModels: " << nModels << endl;
            cerr << "nDatasets: " << nDatasets << endl;
            throw runtime_error("Can't make the Friedman test with less than 3 models and/or less than 3 datasets.");
        }
        ranksModels.clear();
        computeRanks();
        // Set the control model as the one with the lowest average rank
        controlIdx = distance(ranks.begin(), min_element(ranks.begin(), ranks.end(), [](const auto& l, const auto& r) { return l.second < r.second; }));
        computeWTL();
        maxModelName = (*max_element(models.begin(), models.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const string& a, const string& b) { return a.size() < b.size(); })).size();
        fitted = true;
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
    void Statistics::computeRanks()
    {
        map<string, float> ranksLine;
        for (const auto& dataset : datasets) {
            vector<pair<string, double>> ranksOrder;
            for (const auto& model : models) {
                double value = data[model].at(dataset).at(0).get<double>();
                ranksOrder.push_back({ model, value });
            }
            // Assign the ranks
            ranksLine = assignRanks(ranksOrder);
            // Store the ranks of the dataset
            ranksModels[dataset] = ranksLine;
            if (ranks.size() == 0) {
                ranks = ranksLine;
            } else {
                for (const auto& rank : ranksLine) {
                    ranks[rank.first] += rank.second;
                }
            }
        }
        // Average the ranks
        for (const auto& rank : ranks) {
            ranks[rank.first] /= nDatasets;
        }
    }
    void Statistics::computeWTL()
    {
        // Compute the WTL matrix
        for (int i = 0; i < nModels; ++i) {
            wtl[i] = { 0, 0, 0 };
        }
        json origin = data.begin().value();
        for (auto const& item : origin.items()) {
            auto controlModel = models.at(controlIdx);
            double controlValue = data[controlModel].at(item.key()).at(0).get<double>();
            for (int i = 0; i < nModels; ++i) {
                if (i == controlIdx) {
                    continue;
                }
                double value = data[models[i]].at(item.key()).at(0).get<double>();
                if (value < controlValue) {
                    wtl[i].win++;
                } else if (value == controlValue) {
                    wtl[i].tie++;
                } else {
                    wtl[i].loss++;
                }
            }
        }
    }

    void Statistics::postHocHolmTest(bool friedmanResult)
    {
        if (!fitted) {
            fit();
        }
        stringstream oss;
        // Reference https://link.springer.com/article/10.1007/s44196-022-00083-8
        // Post-hoc Holm test
        // Calculate the p-value for the models paired with the control model
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
        holmResult.model = models.at(controlIdx);
        auto color = friedmanResult ? Colors::CYAN() : Colors::YELLOW();
        oss << color;
        oss << "  *************************************************************************************************************" << endl;
        oss << "  Post-hoc Holm test: H0: 'There is no significant differences between the control model and the other models.'" << endl;
        oss << "  Control model: " << models.at(controlIdx) << endl;
        oss << "  " << left << setw(maxModelName) << string("Model") << " p-value      rank      win tie loss Status" << endl;
        oss << "  " << string(maxModelName, '=') << " ============ ========= === === ==== =============" << endl;
        // sort ranks from lowest to highest
        vector<pair<string, float>> ranksOrder;
        for (const auto& rank : ranks) {
            ranksOrder.push_back({ rank.first, rank.second });
        }
        sort(ranksOrder.begin(), ranksOrder.end(), [](const pair<string, float>& a, const pair<string, float>& b) {
            return a.second < b.second;
            });
        // Show the control model info.
        oss << "  " << Colors::BLUE() << left << setw(maxModelName) << ranksOrder.at(0).first << " ";
        oss << setw(12) << " " << setprecision(7) << fixed << " " << ranksOrder.at(0).second << endl;
        for (const auto& item : ranksOrder) {
            auto idx = distance(models.begin(), find(models.begin(), models.end(), item.first));
            double pvalue = 0.0;
            for (const auto& stat : statsOrder) {
                if (stat.first == idx) {
                    pvalue = stat.second;
                }
            }
            holmResult.holmLines.push_back({ item.first, pvalue, item.second, wtl.at(idx), pvalue < significance });
            if (item.first == models.at(controlIdx)) {
                continue;
            }
            auto colorStatus = pvalue > significance ? Colors::GREEN() : Colors::MAGENTA();
            auto status = pvalue > significance ? Symbols::check_mark : Symbols::cross;
            auto textStatus = pvalue > significance ? " accepted H0" : " rejected H0";
            oss << "  " << colorStatus << left << setw(maxModelName) << item.first << " ";
            oss << setprecision(6) << scientific << pvalue << setprecision(7) << fixed << " " << item.second;
            oss << " " << right << setw(3) << wtl.at(idx).win << " " << setw(3) << wtl.at(idx).tie << " " << setw(4) << wtl.at(idx).loss;
            oss << " " << status << textStatus << endl;
        }
        oss << color << "  *************************************************************************************************************" << endl;
        oss << Colors::RESET();
        if (output) {
            cout << oss.str();
        }
    }
    bool Statistics::friedmanTest()
    {
        if (!fitted) {
            fit();
        }
        stringstream oss;
        // Friedman test
        // Calculate the Friedman statistic
        oss << Colors::BLUE() << endl;
        oss << "***************************************************************************************************************" << endl;
        oss << Colors::GREEN() << "Friedman test: H0: 'There is no significant differences between all the classifiers.'" << Colors::BLUE() << endl;
        double degreesOfFreedom = nModels - 1.0;
        double sumSquared = 0;
        for (const auto& rank : ranks) {
            sumSquared += pow(rank.second, 2);
        }
        // Compute the Friedman statistic as in https://link.springer.com/article/10.1007/s44196-022-00083-8
        double friedmanQ = 12.0 * nDatasets / (nModels * (nModels + 1)) * (sumSquared - (nModels * pow(nModels + 1, 2)) / 4);
        // Calculate the critical value
        boost::math::chi_squared chiSquared(degreesOfFreedom);
        long double p_value = (long double)1.0 - cdf(chiSquared, friedmanQ);
        double criticalValue = quantile(chiSquared, 1 - significance);
        oss << "Friedman statistic: " << friedmanQ << endl;
        oss << "Critical χ2 Value for df=" << fixed << (int)degreesOfFreedom
            << " and alpha=" << setprecision(2) << fixed << significance << ": " << setprecision(7) << scientific << criticalValue << std::endl;
        oss << "p-value: " << scientific << p_value << " is " << (p_value < significance ? "less" : "greater") << " than " << setprecision(2) << fixed << significance << endl;
        bool result;
        if (p_value < significance) {
            oss << Colors::GREEN() << "The null hypothesis H0 is rejected." << endl;
            result = true;
        } else {
            oss << Colors::YELLOW() << "The null hypothesis H0 is accepted. Computed p-values will not be significant." << endl;
            result = false;
        }
        oss << Colors::BLUE() << "***************************************************************************************************************" << Colors::RESET() << endl;
        if (output) {
            cout << oss.str();
        }
        friedmanResult = { friedmanQ, criticalValue, p_value, result };
        return result;
    }
    FriedmanResult& Statistics::getFriedmanResult()
    {
        return friedmanResult;
    }
    HolmResult& Statistics::getHolmResult()
    {
        return holmResult;
    }
    map<string, map<string, float>>& Statistics::getRanks()
    {
        return ranksModels;
    }
} // namespace platform
