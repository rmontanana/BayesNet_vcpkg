#include "Folding.h"
#include <map>
#include <iostream>
using namespace std;
class A {
private:
    int a;
public:
    A(int a) : a(a) {}
    int getA() { return a; }
};
class B : public A {
private:
    int b;
public:
    B(int a, int b) : A(a), b(b) {}
    int getB() { return b; }
};
class C : public A {
private:
    int c;
public:
    C(int a, int c) : A(a), c(c) {}
    int getC() { return c; }
};

string counts(vector<int> y, vector<int> indices)
{
    auto result = map<int, int>();
    for (auto i = 0; i < indices.size(); ++i) {
        result[y[indices[i]]]++;
    }
    string final_result = "";
    for (auto i = 0; i < result.size(); ++i)
        final_result += to_string(i) + " -> " + to_string(result[i]) + " // ";
    final_result += "\n";
    return final_result;
}

int main()
{
    auto y = vector<int>(153);
    fill(y.begin(), y.begin() + 50, 0);
    fill(y.begin() + 50, y.begin() + 103, 1);
    fill(y.begin() + 103, y.end(), 2);
    //auto fold = KFold(5, 150);
    auto fold = StratifiedKFold(5, y, -1);
    for (int i = 0; i < 5; ++i) {
        cout << "Fold: " << i << endl;
        auto [train, test] = fold.getFold(i);
        cout << "Train: ";
        cout << "(" << train.size() << "): ";
        for (auto j = 0; j < static_cast<int>(train.size()); j++)
            cout << train[j] << ", ";
        cout << endl;
        cout << "Train Statistics : " << counts(y, train);
        cout << "-------------------------------------------------------------------------------" << endl;
        cout << "Test: ";
        cout << "(" << test.size() << "): ";
        for (auto j = 0; j < static_cast<int>(test.size()); j++)
            cout << test[j] << ", ";
        cout << endl;
        cout << "Test Statistics: " << counts(y, test);
        cout << "==============================================================================" << endl;
        torch::Tensor a = torch::zeros({ 5, 3 });
        torch::Tensor b = torch::zeros({ 5 }) + 1;
        torch::Tensor c = torch::cat({ a,  b.view({5, 1}) }, 1);
        cout << "a:" << a.sizes() << endl;
        cout << a << endl;
        cout << "b:" << b.sizes() << endl;
        cout << b << endl;
        cout << "c:" << c.sizes() << endl;
        cout << c << endl;
        torch::Tensor d = torch::zeros({ 5, 3 });
        torch::Tensor e = torch::tensor({ 1,2,3,4,5 }) + 1;
        torch::Tensor f = torch::cat({ d,  e.view({5, 1}) }, 1);
        cout << "d:" << d.sizes() << endl;
        cout << d << endl;
        cout << "e:" << e.sizes() << endl;
        cout << e << endl;
        cout << "f:" << f.sizes() << endl;
        cout << f << endl;
        auto indices = torch::tensor({ 0, 2, 4 });
        auto k = f.index({ indices, "..." });
        cout << "k:" << k.sizes() << endl;
        cout << k << endl;
        auto w = torch::index_select(f, 0, indices);
        cout << "w:" << w.sizes() << endl;
        cout << w << endl;

        // cout << "Vector poly" << endl;
            // auto some = vector<A>();
            // auto cx = C(5, 4);
            // auto bx = B(7, 6);
            // some.push_back(cx);
            // some.push_back(bx);
            // for (auto& obj : some) {
            // 	cout << "Obj :" << obj.getA() << endl;
            // }
    }
}
