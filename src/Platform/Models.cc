#include "Models.h"
namespace platform {
    using namespace std;
    map<string, bayesnet::BaseClassifier*> Models::classifiers = map<string, bayesnet::BaseClassifier*>({
        { "AODE", new bayesnet::AODE() }, { "KDB", new bayesnet::KDB(2) },
        { "SPODE",  new bayesnet::SPODE(2) }, { "TAN",  new bayesnet::TAN() }
        });
}