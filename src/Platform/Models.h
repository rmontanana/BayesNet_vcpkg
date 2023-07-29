#ifndef MODELS_H
#define MODELS_H
#include <map>
#include "BaseClassifier.h"
#include "AODE.h"
#include "TAN.h"
#include "KDB.h"
#include "SPODE.h"
namespace platform {
    class Models {
    private:
        static map<string, bayesnet::BaseClassifier*> classifiers;
    public:
        static bayesnet::BaseClassifier* get(string name) { return classifiers[name]; }
        static vector<string> getNames()
        {
            vector<string> names;
            for (auto& [name, classifier] : classifiers) {
                names.push_back(name);
            }
            return names;
        }
        static string toString()
        {
            string names = "";
            for (auto& [name, classifier] : classifiers) {
                names += name + ", ";
            }
            return "{" + names.substr(0, names.size() - 2) + "}";
        }
    };
}
#endif