#ifndef MODEL_REGISTER_H
#define MODEL_REGISTER_H
static platform::Registrar registrarT("TAN",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TAN();});
static platform::Registrar registrarTN("TANLd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::TANLd();});
static platform::Registrar registrarS("SPODE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODE(2);});
static platform::Registrar registrarSN("SPODELd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::SPODELd(2);});
static platform::Registrar registrarK("KDB",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDB(2);});
static platform::Registrar registrarKN("KDBLd",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::KDBLd(2);});
static platform::Registrar registrarA("AODE",
    [](void) -> bayesnet::BaseClassifier* { return new bayesnet::AODE();});
#endif