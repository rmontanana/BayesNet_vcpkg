#ifndef PYCLF_H
#define PYCLF_H
#include <string>
#include "DotEnv.h"
namespace PyClassifiers {
    class PyClf {
    public:
        PyClf(const std::string& name);
        virtual ~PyClf();
    private:
        std::string name;

    };
} /* namespace PyClassifiers */
#endif /* PYCLF_H */