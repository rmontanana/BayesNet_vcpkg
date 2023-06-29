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
    string Node::getCPDKey(const Node* child) const
    {
        return name + "-" + child->getName();
    }
}