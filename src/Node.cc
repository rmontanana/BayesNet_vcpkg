#include "Node.h"

namespace bayesnet {
    int Node::next_id = 0;

    Node::Node(const std::string& name, int numStates)
        : id(next_id++), name(name), numStates(numStates), cpt(torch::Tensor()), parents(vector<Node*>()), children(vector<Node*>())
    {
    }

    string Node::getName() const
    {
        return name;
    }

    void Node::addParent(Node* parent)
    {
        parents.push_back(parent);
    }
    void Node::removeParent(Node* parent)
    {
        parents.erase(std::remove(parents.begin(), parents.end(), parent), parents.end());
    }
    void Node::removeChild(Node* child)
    {
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }
    void Node::addChild(Node* child)
    {
        children.push_back(child);
    }
    vector<Node*>& Node::getParents()
    {
        return parents;
    }
    vector<Node*>& Node::getChildren()
    {
        return children;
    }
    int Node::getNumStates() const
    {
        return numStates;
    }
    void Node::setNumStates(int numStates)
    {
        this->numStates = numStates;
    }
    torch::Tensor& Node::getCPT()
    {
        return cpt;
    }
    void Node::setCPT(const torch::Tensor& cpt)
    {
        this->cpt = cpt;
    }
    /*
     The MinFill criterion is a heuristic for variable elimination.
     The variable that minimizes the number of edges that need to be added to the graph to make it triangulated.
     This is done by counting the number of edges that need to be added to the graph if the variable is eliminated.
     The variable with the minimum number of edges is chosen.
     Here this is done computing the length of the combinations of the node neighbors taken 2 by 2.
    */
    unsigned Node::minFill()
    {
        set<string> neighbors;
        for (auto child : children) {
            neighbors.emplace(child->getName());
        }
        for (auto parent : parents) {
            neighbors.emplace(parent->getName());
        }
        return combinations(neighbors).size();
    }
    vector<string> Node::combinations(const set<string>& neighbors)
    {
        vector<string> source(neighbors.begin(), neighbors.end());
        vector<string> result;
        for (int i = 0; i < source.size(); ++i) {
            string temp = source[i];
            for (int j = i + 1; j < source.size(); ++j) {
                result.push_back(temp + source[j]);
            }
        }
        return result;
    }
    Factor* Node::toFactor()
    {
        vector<string> variables;
        vector<int> cardinalities;
        variables.push_back(name);
        cardinalities.push_back(numStates);
        for (auto parent : parents) {
            variables.push_back(parent->getName());
            cardinalities.push_back(parent->getNumStates());
        }
        return new Factor(variables, cardinalities, cpt);

    }
}