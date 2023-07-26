#include "ArffFiles.h"
#include <fstream>
#include <sstream>
#include <map>

using namespace std;

ArffFiles::ArffFiles() = default;

vector<string> ArffFiles::getLines() const
{
    return lines;
}

unsigned long int ArffFiles::getSize() const
{
    return lines.size();
}

vector<pair<string, string>> ArffFiles::getAttributes() const
{
    return attributes;
}

string ArffFiles::getClassName() const
{
    return className;
}

string ArffFiles::getClassType() const
{
    return classType;
}

vector<vector<float>>& ArffFiles::getX()
{
    return X;
}

vector<int>& ArffFiles::getY()
{
    return y;
}

void ArffFiles::loadCommon(string fileName)
{
    ifstream file(fileName);
    if (!file.is_open()) {
        throw invalid_argument("Unable to open file");
    }
    string line;
    string keyword;
    string attribute;
    string type;
    string type_w;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%' || line == "\r" || line == " ") {
            continue;
        }
        if (line.find("@attribute") != string::npos || line.find("@ATTRIBUTE") != string::npos) {
            stringstream ss(line);
            ss >> keyword >> attribute;
            type = "";
            while (ss >> type_w)
                type += type_w + " ";
            attributes.emplace_back(trim(attribute), trim(type));
            continue;
        }
        if (line[0] == '@') {
            continue;
        }
        lines.push_back(line);
    }
    file.close();
    if (attributes.empty())
        throw invalid_argument("No attributes found");
}

void ArffFiles::load(const string& fileName, bool classLast)
{
    int labelIndex;
    loadCommon(fileName);
    if (classLast) {
        className = get<0>(attributes.back());
        classType = get<1>(attributes.back());
        attributes.pop_back();
        labelIndex = static_cast<int>(attributes.size());
    } else {
        className = get<0>(attributes.front());
        classType = get<1>(attributes.front());
        attributes.erase(attributes.begin());
        labelIndex = 0;
    }
    generateDataset(labelIndex);
}
void ArffFiles::load(const string& fileName, const string& name)
{
    int labelIndex;
    loadCommon(fileName);
    bool found = false;
    for (int i = 0; i < attributes.size(); ++i) {
        if (attributes[i].first == name) {
            className = get<0>(attributes[i]);
            classType = get<1>(attributes[i]);
            attributes.erase(attributes.begin() + i);
            labelIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        throw invalid_argument("Class name not found");
    }
    generateDataset(labelIndex);
}

void ArffFiles::generateDataset(int labelIndex)
{
    X = vector<vector<float>>(attributes.size(), vector<float>(lines.size()));
    auto yy = vector<string>(lines.size(), "");
    for (size_t i = 0; i < lines.size(); i++) {
        stringstream ss(lines[i]);
        string value;
        int pos = 0;
        int xIndex = 0;
        while (getline(ss, value, ',')) {
            if (pos++ == labelIndex) {
                yy[i] = value;
            } else {
                X[xIndex++][i] = stof(value);
            }
        }
    }
    y = factorize(yy);
}

string ArffFiles::trim(const string& source)
{
    string s(source);
    s.erase(0, s.find_first_not_of(" '\n\r\t"));
    s.erase(s.find_last_not_of(" '\n\r\t") + 1);
    return s;
}

vector<int> ArffFiles::factorize(const vector<string>& labels_t)
{
    vector<int> yy;
    yy.reserve(labels_t.size());
    map<string, int> labelMap;
    int i = 0;
    for (const string& label : labels_t) {
        if (labelMap.find(label) == labelMap.end()) {
            labelMap[label] = i++;
        }
        yy.push_back(labelMap[label]);
    }
    return yy;
}