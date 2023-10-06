#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "TestUtils.h"
#include "Folding.h"

TEST_CASE("KFold Test", "[KFold]")
{
    // Initialize a KFold object with k=5 and a seed of 19.
    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto raw = RawDatasets(file_name, true);
    int nFolds = 5;
    platform::KFold kfold(nFolds, raw.nSamples, 19);s
        int number = raw.nSamples * (kfold.getNumberOfFolds() - 1) / kfold.getNumberOfFolds();

    SECTION("Number of Folds")
    {
        REQUIRE(kfold.getNumberOfFolds() == nFolds);
    }
    SECTION("Fold Test")
    {
        // Test each fold's size and contents.
        for (int i = 0; i < nFolds; ++i) {
            auto [train_indices, test_indices] = kfold.getFold(i);
            bool result = train_indices.size() == number || train_indices.size() == number + 1;
            REQUIRE(result);
            REQUIRE(train_indices.size() + test_indices.size() == raw.nSamples);
        }
    }
}

map<int, int> counts(vector<int> y, vector<int> indices)
{
    map<int, int> result;
    for (auto i = 0; i < indices.size(); ++i) {
        result[y[indices[i]]]++;
    }
    return result;
}

TEST_CASE("StratifiedKFold Test", "[StratifiedKFold]")
{
    int nFolds = 3;
    // Initialize a StratifiedKFold object with k=3, using the y vector, and a seed of 17.
    string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto raw = RawDatasets(file_name, true);
    platform::StratifiedKFold stratified_kfoldt(nFolds, raw.yt, 17);
    platform::StratifiedKFold stratified_kfoldv(nFolds, raw.yv, 17);
    int number = raw.nSamples * (stratified_kfold.getNumberOfFolds() - 1) / stratified_kfold.getNumberOfFolds();

    // SECTION("Number of Folds")
    // {
    //     REQUIRE(stratified_kfold.getNumberOfFolds() == nFolds);
    // }
    SECTION("Fold Test")
    {
        // Test each fold's size and contents.
        auto counts = vector<int>(raw.classNumStates, 0);
        for (int i = 0; i < nFolds; ++i) {
            auto [train_indicest, test_indicest] = stratified_kfoldt.getFold(i);
            auto [train_indicesv, test_indicesv] = stratified_kfoldv.getFold(i);
            REQUIRE(train_indicest == train_indicesv);
            REQUIRE(test_indicest == test_indicesv);

            bool result = train_indices.size() == number || train_indices.size() == number + 1;
            REQUIRE(result);
            REQUIRE(train_indices.size() + test_indices.size() == raw.nSamples);
            auto train_t = torch::tensor(train_indices);
            auto ytrain = raw.yt.index({ train_t });
            cout << "dataset=" << file_name << endl;
            cout << "nSamples=" << raw.nSamples << endl;;
            cout << "number=" << number << endl;
            cout << "train_indices.size()=" << train_indices.size() << endl;
            cout << "test_indices.size()=" << test_indices.size() << endl;
            cout << "Class Name = " << raw.classNamet << endl;
            cout << "Features = ";
            for (const auto& item : raw.featurest) {
                cout << item << ", ";
            }
            cout << endl;
            cout << "Class States: ";
            for (const auto& item : raw.statest.at(raw.classNamet)) {
                cout << item << ", ";
            }
            cout << endl;
            // Check that the class labels have been equally assign to each fold
            for (const auto& idx : train_indices) {
                counts[ytrain[idx].item<int>()]++;
            }
            int j = 0;
            for (const auto& item : counts) {
                cout << "j=" << j++ << item << endl;
            }

        }
        REQUIRE(1 == 1);
    }
}
