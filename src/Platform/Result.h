#ifndef RESULT_H
#define RESULT_H
#include <string>
#include <chrono>

using namespace std;
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
    float score, score_std, train_time, train_time_std, test_time, test_time_std;
public:
    Result() = default;
    Result& setDataset(string dataset) { this->dataset = dataset; return *this; }
    Result& setHyperparameters(string hyperparameters) { this->hyperparameters = hyperparameters; return *this; }
    Result& setSamples(int samples) { this->samples = samples; return *this; }
    Result& setFeatures(int features) { this->features = features; return *this; }
    Result& setClasses(int classes) { this->classes = classes; return *this; }
    Result& setScore(float score) { this->score = score; return *this; }
    Result& setScoreStd(float score_std) { this->score_std = score_std; return *this; }
    Result& setTrainTime(float train_time) { this->train_time = train_time; return *this; }
    Result& setTrainTimeStd(float train_time_std) { this->train_time_std = train_time_std; return *this; }
    Result& setTestTime(float test_time) { this->test_time = test_time; return *this; }
    Result& setTestTimeStd(float test_time_std) { this->test_time_std = test_time_std; return *this; }
};
class Experiment {
private:
    string title, model, platform, score_name, model_version, language_version;
    bool discretized, stratified;
    vector<Result> results;
    vector<int> random_seeds;
    int nfolds;
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
    void save(string path) { cout << "Saving experiment..." << endl; }
};
#endif