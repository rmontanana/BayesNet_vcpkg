#ifndef NODE_H
#define NODE_H
#include <torch/torch.h>
#include "Factor.h"
#include <vector>
#include <string>
namespace bayesnet {
    using namespace std;
    class Node {
    private:
        static int next_id;
        const int id;
        string name;
        vector<Node*> parents;
        vector<Node*> children;
        torch::Tensor cpTable;
        int numStates;
        torch::Tensor cpt;
        vector<string> combinations(const set<string>&);
    public:
        Node(const std::string&, int);
        void addParent(Node*);
        void addChild(Node*);
        void removeParent(Node*);
        void removeChild(Node*);
        string getName() const;
        vector<Node*>& getParents();
        vector<Node*>& getChildren();
        torch::Tensor& getCPT();
        void setCPT(const torch::Tensor&);
        int getNumStates() const;
        void setNumStates(int);
        unsigned minFill();
        int getId() const { return id; }
        Factor* toFactor();
    };
}
#endif