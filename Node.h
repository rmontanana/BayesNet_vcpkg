#ifndef NODE_H
#define NODE_H
#include <torch/torch.h>
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
        int numStates;
        torch::Tensor cpt;
    public:
        Node(const std::string& name, int numStates);
        void addParent(Node* parent);
        void addChild(Node* child);
        string getName() const;
        vector<Node*>& getParents();
        vector<Node*>& getChildren();
        torch::Tensor& getCPT();
        void setCPT(const torch::Tensor& cpt);
        int getNumStates() const;
        int getId() const { return id; }
        string getCPDKey(const Node*) const;
    };
}
#endif