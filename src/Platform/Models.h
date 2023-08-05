#ifndef MODELS_H
#define MODELS_H
#include <map>
#include "BaseClassifier.h"
#include "AODE.h"
#include "TAN.h"
#include "KDB.h"
#include "SPODE.h"
#include "TANNew.h"
#include "KDBNew.h"
namespace platform {
    class Models {
    private:
        map<string, function<bayesnet::BaseClassifier* (void)>> functionRegistry;
        static Models* factory; //singleton
        Models() {};
    public:
        Models(Models&) = delete;
        void operator=(const Models&) = delete;
        // Idea from: https://www.codeproject.com/Articles/567242/AplusC-2b-2bplusObjectplusFactory
        static Models* instance();
        shared_ptr<bayesnet::BaseClassifier> create(const string& name);
        void registerFactoryFunction(const string& name,
            function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
        vector<string> getNames();
        string toString();

    };
    class Registrar {
    public:
        Registrar(const string& className, function<bayesnet::BaseClassifier* (void)> classFactoryFunction);
    };
}
#endif