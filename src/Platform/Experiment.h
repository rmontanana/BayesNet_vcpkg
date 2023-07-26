#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include "Folding.h"
#include "BaseClassifier.h"
#include "TAN.h"
#include "KDB.h"
#include "AODE.h"

using namespace std;
namespace platform {
    using json = nlohmann::json;
    class Timer {
    private:
        chrono::time_point<chrono::steady_clock> begin;
    public:
        Timer() = default;
        ~Timer() = default;
        void start() { begin = chrono::high_resolution_clock::now(); }
        float getDuration() { return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - begin).count(); }
    };
    class Result {
    private:
        string dataset, hyperparameters;
        int samples, features, classes;
        float score_train, score_test, score_train_std, score_test_std, train_time, train_time_std, test_time, test_time_std;
    public:
        Result() = default;
        Result& setDataset(string dataset) { this->dataset = dataset; return *this; }
        Result& setHyperparameters(string hyperparameters) { this->hyperparameters = hyperparameters; return *this; }
        Result& setSamples(int samples) { this->samples = samples; return *this; }
        Result& setFeatures(int features) { this->features = features; return *this; }
        Result& setClasses(int classes) { this->classes = classes; return *this; }
        Result& setScoreTrain(float score) { this->score_train = score; return *this; }
        Result& setScoreTest(float score) { this->score_test = score; return *this; }
        Result& setScoreTrainStd(float score_std) { this->score_train_std = score_std; return *this; }
        Result& setScoreTestStd(float score_std) { this->score_test_std = score_std; return *this; }
        Result& setTrainTime(float train_time) { this->train_time = train_time; return *this; }
        Result& setTrainTimeStd(float train_time_std) { this->train_time_std = train_time_std; return *this; }
        Result& setTestTime(float test_time) { this->test_time = test_time; return *this; }
        Result& setTestTimeStd(float test_time_std) { this->test_time_std = test_time_std; return *this; }
        const float get_score_train() const { return score_train; }
        float get_score_test() { return score_test; }
        const string& getDataset() const { return dataset; }
        const string& getHyperparameters() const { return hyperparameters; }
        const int getSamples() const { return samples; }
        const int getFeatures() const { return features; }
        const int getClasses() const { return classes; }
        const float getScoreTrain() const { return score_train; }
        const float getScoreTest() const { return score_test; }
        const float getScoreTrainStd() const { return score_train_std; }
        const float getScoreTestStd() const { return score_test_std; }
        const float getTrainTime() const { return train_time; }
        const float getTrainTimeStd() const { return train_time_std; }
        const float getTestTime() const { return test_time; }
        const float getTestTimeStd() const { return test_time_std; }
    };
    class Experiment {
    private:
        string title, model, platform, score_name, model_version, language_version;
        bool discretized, stratified;
        vector<Result> results;
        vector<int> random_seeds;
        int nfolds;
        float duration;
        json build_json();
    public:
        Experiment() = default;
        Experiment& setTitle(string title) { this->title = title; return *this; }
        Experiment& setModel(string model) { this->model = model; return *this; }
        Experiment& setPlatform(string platform) { this->platform = platform; return *this; }
        Experiment& setScoreName(string score_name) { this->score_name = score_name; return *this; }
        Experiment& setModelVersion(string model_version) { this->model_version = model_version; return *this; }
        Experiment& setLanguageVersion(string language_version) { this->language_version = language_version; return *this; }
        Experiment& setDiscretized(bool discretized) { this->discretized = discretized; return *this; }
        Experiment& setStratified(bool stratified) { this->stratified = stratified; return *this; }
        Experiment& setNFolds(int nfolds) { this->nfolds = nfolds; return *this; }
        Experiment& addResult(Result result) { results.push_back(result); return *this; }
        Experiment& addRandomSeed(int random_seed) { random_seeds.push_back(random_seed); return *this; }
        Experiment& setDuration(float duration) { this->duration = duration; return *this; }
        string get_file_name();
        void save(string path);
        void show() { cout << "Showing experiment..." << "Score Test: " << results[0].get_score_test() << " Score Train: " << results[0].get_score_train() << endl; }
    };
    Result cross_validation(Fold* fold, string model_name, torch::Tensor& X, torch::Tensor& y, vector<string> features, string className, map<string, vector<int>> states);
}
#endif