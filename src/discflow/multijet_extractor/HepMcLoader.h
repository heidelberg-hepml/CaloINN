#ifndef INCLUDE_GUARD_24A1A100_0ADB_4705_AED5_F08D6EAA6E75
#define INCLUDE_GUARD_24A1A100_0ADB_4705_AED5_F08D6EAA6E75

#include "HepMC/IO_GenEvent.h"

namespace MultijetExtractor {

class HepMcLoader {
private:
    bool hard_process;
    HepMC::IO_GenEvent hepmc_in;
public:
    HepMcLoader(char const* file, bool _hard_process);
    bool next_event(std::vector<HepMC::FourVector>& constits);
};

}

#endif //INCLUDE_GUARD_24A1A100_0ADB_4705_AED5_F08D6EAA6E75
