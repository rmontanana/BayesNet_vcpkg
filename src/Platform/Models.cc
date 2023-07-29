#include "Models.h"
namespace platform {
    using namespace std;
    // map<string, bayesnet::BaseClassifier*> Models::classifiers = map<string, bayesnet::BaseClassifier*>({
    //     { "AODE", new bayesnet::AODE() }, { "KDB", new bayesnet::KDB(2) },
    //     { "SPODE",  new bayesnet::SPODE(2) }, { "TAN",  new bayesnet::TAN() }
    //     });
    // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
    shared_ptr<bayesnet::BaseClassifier> Models::createInstance(const string& name)
    {
        bayesnet::BaseClassifier* instance = nullptr;
        if (name == "AODE") {
            instance = new bayesnet::AODE();
        } else if (name == "KDB") {
            instance = new bayesnet::KDB(2);
        } else if (name == "SPODE") {
            instance = new bayesnet::SPODE(2);
        } else if (name == "TAN") {
            instance = new bayesnet::TAN();
        } else {
            throw runtime_error("Model " + name + " not found");
        }
        if (instance != nullptr)
            return shared_ptr<bayesnet::BaseClassifier>(instance);
        else
            return nullptr;
    }
}