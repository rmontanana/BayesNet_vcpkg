#ifndef MST_H
#define MST_H
#include <torch/torch.h>
#include <vector>
#include <string>
namespace bayesnet {
    class MST {
    private:
        torch::Tensor weights;
        std::vector<std::string> features;
        int root = 0;
    public:
        MST() = default;
        MST(const std::vector<std::string>& features, const torch::Tensor& weights, const int root);
        std::vector<std::pair<int, int>> maximumSpanningTree();
    };
    class Graph {
    private:
        int V;      // number of nodes in graph
        std::vector <std::pair<float, std::pair<int, int>>> G; // std::vector for graph
        std::vector <std::pair<float, std::pair<int, int>>> T; // std::vector for mst
        std::vector<int> parent;
    public:
        explicit Graph(int V);
        void addEdge(int u, int v, float wt);
        int find_set(int i);
        void union_set(int u, int v);
        void kruskal_algorithm();
        void display_mst();
        std::vector <std::pair<float, std::pair<int, int>>> get_mst() { return T; }
    };
}
#endif