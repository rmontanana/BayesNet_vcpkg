#include "Experiment.h"

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
        result["seeds"] = random_seeds;
        result["duration"] = duration;
        result["results"] = json::array();
        for (auto& r : results) {
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
            j["nodes"] = r.getNodes();
            j["leaves"] = r.getLeaves();
            j["depth"] = r.getDepth();
            result["results"].push_back(j);
        }
        return result;
    }
    void Experiment::save(string path)
    {
        json data = build_json();
        ofstream file(path + get_file_name());
        file << data;
        file.close();
    }
    Result cross_validation(Fold* fold, string model_name, torch::Tensor& Xt, torch::Tensor& y, vector<string> features, string className, map<string, vector<int>> states)
    {
        auto classifiers = map<string, bayesnet::BaseClassifier*>({
           { "AODE", new bayesnet::AODE() }, { "KDB", new bayesnet::KDB(2) },
           { "SPODE",  new bayesnet::SPODE(2) }, { "TAN",  new bayesnet::TAN() }
            }
        );
        auto result = Result();
        auto [values, counts] = at::_unique(y);
        result.setSamples(Xt.size(1)).setFeatures(Xt.size(0)).setClasses(values.size(0));
        auto k = fold->getNumberOfFolds();
        auto accuracy_test = torch::zeros({ k }, torch::kFloat64);
        auto accuracy_train = torch::zeros({ k }, torch::kFloat64);
        auto train_time = torch::zeros({ k }, torch::kFloat64);
        auto test_time = torch::zeros({ k }, torch::kFloat64);
        auto nodes = torch::zeros({ k }, torch::kFloat64);
        auto edges = torch::zeros({ k }, torch::kFloat64);
        auto num_states = torch::zeros({ k }, torch::kFloat64);
        Timer train_timer, test_timer;
        cout << "doing Fold: " << flush;
        for (int i = 0; i < k; i++) {
            bayesnet::BaseClassifier* model = classifiers[model_name];
            result.setModelVersion(model->getVersion());
            train_timer.start();
            auto [train, test] = fold->getFold(i);
            auto train_t = torch::tensor(train);
            auto test_t = torch::tensor(test);
            auto X_train = Xt.index({ "...", train_t });
            auto y_train = y.index({ train_t });
            auto X_test = Xt.index({ "...", test_t });
            auto y_test = y.index({ test_t });
            cout << i + 1 << ", " << flush;
            model->fit(X_train, y_train, features, className, states);
            nodes[i] = model->getNumberOfNodes();
            edges[i] = model->getNumberOfEdges();
            num_states[i] = model->getNumberOfStates();
            train_time[i] = train_timer.getDuration();
            auto accuracy_train_value = model->score(X_train, y_train);
            test_timer.start();
            auto accuracy_test_value = model->score(X_test, y_test);
            test_time[i] = test_timer.getDuration();
            accuracy_train[i] = accuracy_train_value;
            accuracy_test[i] = accuracy_test_value;
        }
        cout << "end." << endl;
        result.setScoreTest(torch::mean(accuracy_test).item<double>()).setScoreTrain(torch::mean(accuracy_train).item<double>());
        result.setScoreTestStd(torch::std(accuracy_test).item<double>()).setScoreTrainStd(torch::std(accuracy_train).item<double>());
        result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
        result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(edges).item<double>()).setDepth(torch::mean(num_states).item<double>());
        return result;
    }
}