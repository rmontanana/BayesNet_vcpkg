#include <map>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char const* argv[])
{
    map<string, int> m;
    m["a"] = 1;
    m["b"] = 2;
    m["c"] = 3;
    if (m.find("b") != m.end()) {
        cout << "Found b" << endl;
    } else {
        cout << "Not found b" << endl;
    }
    // for (auto [key, value] : m) {
    //     cout << key << " " << value << endl;
    // }

    return 0;
}
