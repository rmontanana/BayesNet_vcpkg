#include "PyClf.h"

namespace PyClassifiers {

    PyClf::PyClf(const std::std::string& name) : name(name)
    {
        env = platform::DotEnv();


    }


    PyClf::~PyClf()
    {

    }

} /* namespace PyClassifiers */