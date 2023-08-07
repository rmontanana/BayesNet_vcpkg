#include "AODELd.h"
#include "Models.h"

namespace bayesnet {
    using namespace std;
    AODELd::AODELd() : Ensemble(), Proposal(dataset, features, className) {}
    AODELd& AODELd::fit(torch::Tensor& X_, torch::Tensor& y_, vector<string>& features_, string className_, map<string, vector<int>>& states_)
    {
        // This first part should go in a Classifier method called fit_local_discretization o fit_float...
        features = features_;
        className = className_;
        Xf = X_;
        y = y_;
        // Fills vectors Xv & yv with the data from tensors X_ (discretized) & y
        fit_local_discretization(states, y);
        // We have discretized the input data
        // 1st we need to fit the model to build the normal TAN structure, TAN::fit initializes the base Bayesian network
        Ensemble::fit(dataset, features, className, states);
        return *this;

    }
    void AODELd::buildModel()
    {
        models.clear();
        cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaah!" << endl;
        for (int i = 0; i < features.size(); ++i) {
            models.push_back(Models::instance().create("SPODELd"));
            models[i]->test();
        }
        n_models = models.size();
    }
    void AODELd::trainModel()
    {
        cout << "dataset: " << dataset.sizes() << endl;
        cout << "features: " << features.size() << endl;
        cout << "className: " << className << endl;
        cout << "states: " << states.size() << endl;
        for (const auto& model : models) {
            model->fit(dataset, features, className, states);
            model->test();
        }
    }
    vector<string> AODELd::graph(const string& name) const
    {
        return Ensemble::graph(name);
    }
}