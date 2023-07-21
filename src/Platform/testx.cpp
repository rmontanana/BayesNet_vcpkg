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
	auto y = vector<int>(150);
	fill(y.begin(), y.begin() + 50, 0);
	fill(y.begin() + 50, y.begin() + 100, 1);
	fill(y.begin() + 100, y.end(), 2);
	//auto fold = KFold(5, 150);
	auto fold = StratifiedKFold(5, y, 0);
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
