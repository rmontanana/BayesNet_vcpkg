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
        chrono::high_resolution_clock::time_point begin;
    public:
        Timer() = default;
        ~Timer() = default;
        void start() { begin = chrono::high_resolution_clock::now(); }
        double getDuration()
        {
            chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
            chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(end - begin);
            return time_span.count();
        }
    };
    class Result {
    private:
        string dataset, hyperparameters, model_version;
        int samples, features, classes;
        double score_train, score_test, score_train_std, score_test_std, train_time, train_time_std, test_time, test_time_std;
        float nodes, leaves, depth;
        vector<double> scores_train, scores_test, times_train, times_test;
    public:
        Result() = default;
        Result& setDataset(string dataset) { this->dataset = dataset; return *this; }
        Result& setHyperparameters(string hyperparameters) { this->hyperparameters = hyperparameters; return *this; }
        Result& setSamples(int samples) { this->samples = samples; return *this; }
        Result& setFeatures(int features) { this->features = features; return *this; }
        Result& setClasses(int classes) { this->classes = classes; return *this; }
        Result& setScoreTrain(double score) { this->score_train = score; return *this; }
        Result& setScoreTest(double score) { this->score_test = score; return *this; }
        Result& setScoreTrainStd(double score_std) { this->score_train_std = score_std; return *this; }
        Result& setScoreTestStd(double score_std) { this->score_test_std = score_std; return *this; }
        Result& setTrainTime(double train_time) { this->train_time = train_time; return *this; }
        Result& setTrainTimeStd(double train_time_std) { this->train_time_std = train_time_std; return *this; }
        Result& setTestTime(double test_time) { this->test_time = test_time; return *this; }
        Result& setTestTimeStd(double test_time_std) { this->test_time_std = test_time_std; return *this; }
        Result& setNodes(float nodes) { this->nodes = nodes; return *this; }
        Result& setLeaves(float leaves) { this->leaves = leaves; return *this; }
        Result& setDepth(float depth) { this->depth = depth; return *this; }
        Result& addScoreTrain(double score) { scores_train.push_back(score); return *this; }
        Result& addScoreTest(double score) { scores_test.push_back(score); return *this; }
        Result& addTimeTrain(double time) { times_train.push_back(time); return *this; }
        Result& addTimeTest(double time) { times_test.push_back(time); return *this; }
        const float get_score_train() const { return score_train; }
        float get_score_test() { return score_test; }
        const string& getDataset() const { return dataset; }
        const string& getHyperparameters() const { return hyperparameters; }
        const int getSamples() const { return samples; }
        const int getFeatures() const { return features; }
        const int getClasses() const { return classes; }
        const double getScoreTrain() const { return score_train; }
        const double getScoreTest() const { return score_test; }
        const double getScoreTrainStd() const { return score_train_std; }
        const double getScoreTestStd() const { return score_test_std; }
        const double getTrainTime() const { return train_time; }
        const double getTrainTimeStd() const { return train_time_std; }
        const double getTestTime() const { return test_time; }
        const double getTestTimeStd() const { return test_time_std; }
        const float getNodes() const { return nodes; }
        const float getLeaves() const { return leaves; }
        const float getDepth() const { return depth; }
        const vector<double>& getScoresTrain() const { return scores_train; }
        const vector<double>& getScoresTest() const { return scores_test; }
        const vector<double>& getTimesTrain() const { return times_train; }
        const vector<double>& getTimesTest() const { return times_test; }
    };
    class Experiment {
    private:
        string title, model, platform, score_name, model_version, language_version, language;
        bool discretized, stratified;
        vector<Result> results;
        vector<int> randomSeeds;
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
        Experiment& setLanguage(string language) { this->language = language; return *this; }
        Experiment& setLanguageVersion(string language_version) { this->language_version = language_version; return *this; }
        Experiment& setDiscretized(bool discretized) { this->discretized = discretized; return *this; }
        Experiment& setStratified(bool stratified) { this->stratified = stratified; return *this; }
        Experiment& setNFolds(int nfolds) { this->nfolds = nfolds; return *this; }
        Experiment& addResult(Result result) { results.push_back(result); return *this; }
        Experiment& addRandomSeed(int randomSeed) { randomSeeds.push_back(randomSeed); return *this; }
        Experiment& setDuration(float duration) { this->duration = duration; return *this; }
        string get_file_name();
        void save(string path);
        //Result cross_validation(const string& path, const string& fileName);
        Result cross_validation(string model_name, torch::Tensor& X, torch::Tensor& y, vector<string> features, string className, map<string, vector<int>> states);
        void show();
    };
}
#endif