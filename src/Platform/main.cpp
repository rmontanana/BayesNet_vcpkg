#include "Folding.h"
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
int main()
{
	auto fold = KFold(5, 100, 1);
	for (int i = 0; i < 5; ++i) {
		cout << "Fold: " << i << endl;
		auto [train, test] = fold.getFold(i);
		cout << "Train: ";
		cout << "(" << train.size() << "): ";
		for (auto j = 0; j < static_cast<int>(train.size()); j++)
			cout << train[j] << ", ";
		cout << endl;
		cout << "Test: ";
		cout << "(" << train.size() << "): ";
		for (auto j = 0; j < static_cast<int>(test.size()); j++)
			cout << test[j] << ", ";
		cout << endl;
		cout << "Vector poly" << endl;
		auto some = vector<A>();
		auto cx = C(5, 4);
		auto bx = B(7, 6);
		some.push_back(cx);
		some.push_back(bx);
		for (auto& obj : some) {
			cout << "Obj :" << obj.getA() << endl;
		}
	}
}
