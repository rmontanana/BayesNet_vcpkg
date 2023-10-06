#include "Folding.h"
#include "map"
#include "Datasets.h"
#include <map>
#include <iostream>
#include <sstream>
using namespace std;
using namespace platform;

string counts(vector<int> y, vector<int> indices)
{
    auto result = map<int, int>();
    stringstream oss;
    for (auto i = 0; i < indices.size(); ++i) {
        result[y[indices[i]]]++;
    }
    string final_result = "";
    for (auto i = 0; i < result.size(); ++i)
        oss << i << " -> " << setprecision(2) << fixed
        << (double)result[i] * 100 / indices.size() << "% (" << result[i] << ") //";
    oss << endl;
    return oss.str();
}

int main()
{
    map<string, string> balance = {
        {"iris", "33,33% (50) / 33,33% (50) / 33,33% (50)"},
        {"diabetes", "34,90% (268) / 65,10% (500)"},
        {"ecoli", "42,56% (143) / 22,92% (77) / 0,60% (2) / 0,60% (2) / 10,42% (35) / 5,95% (20) / 1,49% (5) / 15,48% (52)"},
        {"glass", "32,71% (70) / 7,94% (17) / 4,21% (9) / 35,51% (76) / 13,55% (29) / 6,07% (13)"}
    };
    for (const auto& file_name : { "iris", "glass", "ecoli", "diabetes" }) {
        auto dt = Datasets(true, "Arff");
        auto [X, y] = dt.getVectors(file_name);
        //auto fold = KFold(5, 150);
        auto fold = StratifiedKFold(5, y, -1);
        cout << "***********************************************************************************************" << endl;
        cout << "Dataset: " << file_name << endl;
        cout << "Nº Samples: " << dt.getNSamples(file_name) << endl;
        cout << "Class states: " << dt.getNClasses(file_name) << endl;
        cout << "Balance: " << balance.at(file_name) << endl;
        for (int i = 0; i < 5; ++i) {
            cout << "Fold: " << i << endl;
            auto [train, test] = fold.getFold(i);
            cout << "Train: ";
            cout << "(" << train.size() << "): ";
            // for (auto j = 0; j < static_cast<int>(train.size()); j++)
            //     cout << train[j] << ", ";
            cout << endl;
            cout << "Train Statistics : " << counts(y, train);
            cout << "-------------------------------------------------------------------------------" << endl;
            cout << "Test: ";
            cout << "(" << test.size() << "): ";
            // for (auto j = 0; j < static_cast<int>(test.size()); j++)
            //     cout << test[j] << ", ";
            cout << endl;
            cout << "Test Statistics: " << counts(y, test);
            cout << "==============================================================================" << endl;
        }
        cout << "***********************************************************************************************" << endl;
    }

}

