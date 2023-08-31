#include "Experiment.h"
#include "Datasets.h"
#include "Models.h"
#include "ReportConsole.h"
#include <fstream>
namespace platform {
    using json = nlohmann::json;
    string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    Experiment::Experiment() : hyperparameters(json::parse("{}")) {}
    string Experiment::get_file_name()
    {
        string result = "results_" + score_name + "_" + model + "_" + platform + "_" + get_date() + "_" + get_time() + "_" + (stratified ? "1" : "0") + ".json";
        return result;
    }

    json Experiment::build_json()
    {
        json result;
        result["title"] = title;
        result["date"] = get_date();
        result["time"] = get_time();
        result["model"] = model;
        result["version"] = model_version;
        result["platform"] = platform;
        result["score_name"] = score_name;
        result["language"] = language;
        result["language_version"] = language_version;
        result["discretized"] = discretized;
        result["stratified"] = stratified;
        result["folds"] = nfolds;
        result["seeds"] = randomSeeds;
        result["duration"] = duration;
        result["results"] = json::array();
        for (const auto& r : results) {
            json j;
            j["dataset"] = r.getDataset();
            j["hyperparameters"] = r.getHyperparameters();
            j["samples"] = r.getSamples();
            j["features"] = r.getFeatures();
            j["classes"] = r.getClasses();
            j["score_train"] = r.getScoreTrain();
            j["score_test"] = r.getScoreTest();
            j["score"] = r.getScoreTest();
            j["score_std"] = r.getScoreTestStd();
            j["score_train_std"] = r.getScoreTrainStd();
            j["score_test_std"] = r.getScoreTestStd();
            j["train_time"] = r.getTrainTime();
            j["train_time_std"] = r.getTrainTimeStd();
            j["test_time"] = r.getTestTime();
            j["test_time_std"] = r.getTestTimeStd();
            j["time"] = r.getTestTime() + r.getTrainTime();
            j["time_std"] = r.getTestTimeStd() + r.getTrainTimeStd();
            j["scores_train"] = r.getScoresTrain();
            j["scores_test"] = r.getScoresTest();
            j["times_train"] = r.getTimesTrain();
            j["times_test"] = r.getTimesTest();
            j["nodes"] = r.getNodes();
            j["leaves"] = r.getLeaves();
            j["depth"] = r.getDepth();
            result["results"].push_back(j);
        }
        return result;
    }
    void Experiment::save(const string& path)
    {
        json data = build_json();
        ofstream file(path + "/" + get_file_name());
        file << data;
        file.close();
    }

    void Experiment::report()
    {
        json data = build_json();
        ReportConsole report(data);
        report.show();
    }

    void Experiment::show()
    {
        json data = build_json();
        cout << data.dump(4) << endl;
    }

    void Experiment::go(vector<string> filesToProcess, const string& path)
    {
        cout << "*** Starting experiment: " << title << " ***" << endl;
        for (auto fileName : filesToProcess) {
            cout << "- " << setw(20) << left << fileName << " " << right << flush;
            cross_validation(path, fileName);
            cout << endl;
        }
    }

    void Experiment::cross_validation(const string& path, const string& fileName)
    {
        auto datasets = platform::Datasets(path, discretized, platform::ARFF);
        // Get dataset
        auto [X, y] = datasets.getTensors(fileName);
        auto states = datasets.getStates(fileName);
        auto features = datasets.getFeatures(fileName);
        auto samples = datasets.getNSamples(fileName);
        auto className = datasets.getClassName(fileName);
        cout << " (" << setw(5) << samples << "," << setw(3) << features.size() << ") " << flush;
        // Prepare Result
        auto result = Result();
        auto [values, counts] = at::_unique(y);
        result.setSamples(X.size(1)).setFeatures(X.size(0)).setClasses(values.size(0));
        result.setHyperparameters(hyperparameters);
        // Initialize results vectors
        int nResults = nfolds * static_cast<int>(randomSeeds.size());
        auto accuracy_test = torch::zeros({ nResults }, torch::kFloat64);
        auto accuracy_train = torch::zeros({ nResults }, torch::kFloat64);
        auto train_time = torch::zeros({ nResults }, torch::kFloat64);
        auto test_time = torch::zeros({ nResults }, torch::kFloat64);
        auto nodes = torch::zeros({ nResults }, torch::kFloat64);
        auto edges = torch::zeros({ nResults }, torch::kFloat64);
        auto num_states = torch::zeros({ nResults }, torch::kFloat64);
        Timer train_timer, test_timer;
        int item = 0;
        for (auto seed : randomSeeds) {
            cout << "(" << seed << ") doing Fold: " << flush;
            Fold* fold;
            if (stratified)
                fold = new StratifiedKFold(nfolds, y, seed);
            else
                fold = new KFold(nfolds, y.size(0), seed);
            for (int nfold = 0; nfold < nfolds; nfold++) {
                auto clf = Models::instance()->create(model);
                setModelVersion(clf->getVersion());
                if (hyperparameters.size() != 0) {
                    clf->setHyperparameters(hyperparameters);
                }
                // Split train - test dataset
                train_timer.start();
                auto [train, test] = fold->getFold(nfold);
                auto train_t = torch::tensor(train);
                auto test_t = torch::tensor(test);
                auto X_train = X.index({ "...", train_t });
                auto y_train = y.index({ train_t });
                auto X_test = X.index({ "...", test_t });
                auto y_test = y.index({ test_t });
                cout << nfold + 1 << ", " << flush;
                // Train model
                clf->fit(X_train, y_train, features, className, states);
                nodes[item] = clf->getNumberOfNodes();
                edges[item] = clf->getNumberOfEdges();
                num_states[item] = clf->getNumberOfStates();
                train_time[item] = train_timer.getDuration();
                auto accuracy_train_value = clf->score(X_train, y_train);
                // Test model
                test_timer.start();
                auto accuracy_test_value = clf->score(X_test, y_test);
                test_time[item] = test_timer.getDuration();
                accuracy_train[item] = accuracy_train_value;
                accuracy_test[item] = accuracy_test_value;
                // Store results and times in vector
                result.addScoreTrain(accuracy_train_value);
                result.addScoreTest(accuracy_test_value);
                result.addTimeTrain(train_time[item].item<double>());
                result.addTimeTest(test_time[item].item<double>());
                item++;
                clf.reset();
            }
            cout << "end. " << flush;
        }
        result.setScoreTest(torch::mean(accuracy_test).item<double>()).setScoreTrain(torch::mean(accuracy_train).item<double>());
        result.setScoreTestStd(torch::std(accuracy_test).item<double>()).setScoreTrainStd(torch::std(accuracy_train).item<double>());
        result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
        result.setTestTimeStd(torch::std(test_time).item<double>()).setTrainTimeStd(torch::std(train_time).item<double>());
        result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(edges).item<double>()).setDepth(torch::mean(num_states).item<double>());
        result.setDataset(fileName);
        addResult(result);
    }
}