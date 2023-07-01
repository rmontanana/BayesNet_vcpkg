#include <iostream>
#include <string>
#include <torch/torch.h>
#include "ArffFiles.h"
#include "Network.h"

using namespace std;

int main()
{
    auto handler = ArffFiles();
    handler.load("iris.arff");
    auto X = handler.getX();
    auto y = handler.getY();
    auto className = handler.getClassName();
    vector<pair<string, string>> edges = { {className, "sepallength"}, {className, "sepalwidth"}, {className, "petallength"}, {className, "petalwidth"} };
    auto network = bayesnet::Network();
    // Add nodes to the network
    for (auto feature : handler.getAttributes()) {
        cout << "Adding feature: " << feature.first << endl;
        network.addNode(feature.first, 7);
    }
    network.addNode(className, 3);
    for (auto item : edges) {
        network.addEdge(item.first, item.second);
    }
    cout << "Hello, Bayesian Networks!" << endl;
    torch::Tensor tensor = torch::eye(3);
    cout << "Now I'll add a cycle" << endl;
    try {
        network.addEdge("petallength", className);
    }
    catch (invalid_argument& e) {
        cout << e.what() << endl;
    }
    cout << tensor << std::endl;
    cout << "Nodes:" << endl;
    for (auto [name, item] : network.getNodes()) {
        cout << "*" << item->getName() << endl;
        cout << "-Parents:" << endl;
        for (auto parent : item->getParents()) {
            cout << " " << parent->getName() << endl;
        }
        cout << "-Children:" << endl;
        for (auto child : item->getChildren()) {
            cout << " " << child->getName() << endl;
        }
    }
    cout << "Root: " << network.getRoot()->getName() << endl;
    network.setRoot(className);
    cout << "Now Root should be class: " << network.getRoot()->getName() << endl;
    cout << "PyTorch version: " << TORCH_VERSION << endl;
    return 0;
}