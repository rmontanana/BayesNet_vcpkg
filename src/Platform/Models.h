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
    public:
        // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
        static shared_ptr<bayesnet::BaseClassifier> createInstance(const string& name);
        static vector<string> getNames()
        {
            return { "aaaaaAODE", "KDB", "SPODE", "TAN" };
        }
        static string toString()
        {
            return "{aaaaae34223AODE, KDB, SPODE, TAN}";
            //return "{" + names.substr(0, names.size() - 2) + "}";
        }
    };
}
#endif