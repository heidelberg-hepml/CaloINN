#ifndef INCLUDE_GUARD_E0A65A82_5DD7_4B37_834C_ACC5900F9EC4
#define INCLUDE_GUARD_E0A65A82_5DD7_4B37_834C_ACC5900F9EC4

//#include <TROOT.h>
//#include <TSystem.h>
//#include <TApplication.h>
//#include <TString.h>
//
//#include <TH2.h>
//#include <THStack.h>
//#include <TLegend.h>
//#include <TPaveText.h>
#include <TClonesArray.h>
//#include <TLorentzVector.h>

//#include <TMath.h>
//#include <Math/Vector3D.h>
//#include <Math/Vector4D.h>

#include <classes/DelphesClasses.h>
#include <ExRootAnalysis/ExRootTreeReader.h>
//#include <ExRootAnalysis/ExRootTreeWriter.h>
#include <ExRootAnalysis/ExRootTreeBranch.h>
//#include <ExRootAnalysis/ExRootUtilities.h>
//#include <ExRootAnalysis/ExRootResult.h>

#include "HepMC/SimpleVector.h"

namespace MultijetExtractor {

class DelphesRootLoader {
private:
    TChain chain;
    ExRootTreeReader* tree_reader;
    int n_entries, current_entry;
    TClonesArray* branch;

public:
    DelphesRootLoader(const char* file);
    ~DelphesRootLoader();
    bool next_event(std::vector<HepMC::FourVector>& constits);
};

}

#endif //INCLUDE_GUARD_E0A65A82_5DD7_4B37_834C_ACC5900F9EC4
