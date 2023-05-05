#ifndef INCLUDE_GUARD_AA3F8CD3_E04E_4B00_BD4C_9501D06A90D4
#define INCLUDE_GUARD_AA3F8CD3_E04E_4B00_BD4C_9501D06A90D4

#include "HepMC/SimpleVector.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/Filter.hh"

namespace MultijetExtractor {

class FastjetFinder {
private:
    fastjet::JetDefinition jet_def;
    fastjet::Selector selector;

public:
    FastjetFinder(double r, double pt_min);
    void process_event(std::vector<HepMC::FourVector>& constits);
};

}

#endif //INCLUDE_GUARD_AA3F8CD3_E04E_4B00_BD4C_9501D06A90D4
