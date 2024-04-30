// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <random>
#include "TestUtils.h"
#include "bayesnet/config.h"

class Paths {
public:
    static std::string datasets()
    {
        return { data_path.begin(), data_path.end() };
    }
};

class ShuffleArffFiles : public ArffFiles {
public:
    ShuffleArffFiles(int num_samples = 0, bool shuffle = false) : ArffFiles(), num_samples(num_samples), shuffle(shuffle) {}
    void load(const std::string& file_name, bool class_last = true)
    {
        ArffFiles::load(file_name, class_last);
        if (num_samples > 0) {
            if (num_samples > getY().size()) {
                throw std::invalid_argument("num_lines must be less than the number of lines in the file");
            }
            auto indices = std::vector<int>(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            if (shuffle) {
                std::mt19937 g{ 173 };
                std::shuffle(indices.begin(), indices.end(), g);
            }
            auto XX = std::vector<std::vector<float>>(attributes.size(), std::vector<float>(num_samples));
            auto yy = std::vector<int>(num_samples);
            for (int i = 0; i < num_samples; i++) {
                yy[i] = getY()[indices[i]];
                for (int j = 0; j < attributes.size(); j++) {
                    XX[j][i] = X[j][indices[i]];
                }
            }
            X = XX;
            y = yy;
        }
    }
private:
    int num_samples;
    bool shuffle;
};

RawDatasets::RawDatasets(const std::string& file_name, bool discretize_, int num_samples_, bool shuffle_, bool class_last, bool debug)
{
    num_samples = num_samples_;
    shuffle = shuffle_;
    discretize = discretize_;
    // Xt can be either discretized or not
    // Xv is always discretized
    loadDataset(file_name, class_last);
    auto yresized = torch::transpose(yt.view({ yt.size(0), 1 }), 0, 1);
    dataset = torch::cat({ Xt, yresized }, 0);
    nSamples = dataset.size(1);
    weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
    weightsv = std::vector<double>(nSamples, 1.0 / nSamples);
    classNumStates = discretize ? states.at(className).size() : 0;
    auto fold = folding::StratifiedKFold(5, yt, 271);
    auto [train, test] = fold.getFold(0);
    auto train_t = torch::tensor(train);
    auto test_t = torch::tensor(test);
    // Get train and validation sets
    X_train = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), train_t });
    y_train = dataset.index({ -1, train_t });
    X_test = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), test_t });
    y_test = dataset.index({ -1, test_t });
    if (debug)
        std::cout << to_string();
}

map<std::string, int> RawDatasets::discretizeDataset(std::vector<mdlp::samples_t>& X)
{

    map<std::string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], yv);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xv.push_back(xd);
    }
    return maxes;
}

void RawDatasets::loadDataset(const std::string& name, bool class_last)
{
    auto handler = ShuffleArffFiles(num_samples, shuffle);
    handler.load(Paths::datasets() + static_cast<std::string>(name) + ".arff", class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    yv = handler.getY();
    // Get className & Features
    className = handler.getClassName();
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    // Discretize Dataset
    auto maxValues = discretizeDataset(X);
    maxValues[className] = *max_element(yv.begin(), yv.end()) + 1;
    if (discretize) {
        // discretize the tensor as well
        Xt = torch::zeros({ static_cast<int>(Xv.size()), static_cast<int>(Xv[0].size()) }, torch::kInt32);
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = std::vector<int>(maxValues[features[i]]);
            iota(begin(states.at(features[i])), end(states.at(features[i])), 0);
            Xt.index_put_({ i, "..." }, torch::tensor(Xv[i], torch::kInt32));
        }
        states[className] = std::vector<int>(maxValues[className]);
        iota(begin(states.at(className)), end(states.at(className)), 0);
    } else {
        Xt = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, torch::kFloat32);
        for (int i = 0; i < features.size(); ++i) {
            Xt.index_put_({ i, "..." }, torch::tensor(X[i]));
        }
    }
    yt = torch::tensor(yv, torch::kInt32);
}

