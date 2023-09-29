#include <iostream>
#include <locale>
#include "Paths.h"
#include "Colors.h"
#include "Datasets.h"
#include "DotEnv.h"

using namespace std;
const int BALANCE_LENGTH = 75;

struct separated : numpunct<char> {
    char do_decimal_point() const { return ','; }
    char do_thousands_sep() const { return '.'; }
    string do_grouping() const { return "\03"; }
};

void outputBalance(const string& balance)
{
    auto temp = string(balance);
    while (temp.size() > BALANCE_LENGTH - 1) {
        auto part = temp.substr(0, BALANCE_LENGTH);
        cout << part << endl;
        cout << setw(48) << " ";
        temp = temp.substr(BALANCE_LENGTH);
    }
    cout << temp << endl;
}

int main(int argc, char** argv)
{
    auto env = platform::DotEnv();
    auto data = platform::Datasets(false, env.get("source_data"));
    locale mylocale(cout.getloc(), new separated);
    locale::global(mylocale);
    cout.imbue(mylocale);
    cout << Colors::GREEN() << "Dataset                        Sampl. Feat. Cls. Balance" << endl;
    string balanceBars = string(BALANCE_LENGTH, '=');
    cout << "============================== ====== ===== === " << balanceBars << endl;
    bool odd = true;
    for (const auto& dataset : data.getNames()) {
        auto color = odd ? Colors::CYAN() : Colors::BLUE();
        cout << color << setw(30) << left << dataset << " ";
        data.loadDataset(dataset);
        auto nSamples = data.getNSamples(dataset);
        cout << setw(6) << right << nSamples << " ";
        cout << setw(5) << right << data.getFeatures(dataset).size() << " ";
        cout << setw(3) << right << data.getNClasses(dataset) << " ";
        stringstream oss;
        string sep = "";
        for (auto number : data.getClassesCounts(dataset)) {
            oss << sep << setprecision(2) << fixed << (float)number / nSamples * 100.0 << "% (" << number << ")";
            sep = " / ";
        }
        outputBalance(oss.str());
        odd = !odd;
    }
    cout << Colors::RESET() << endl;
    return 0;
}
