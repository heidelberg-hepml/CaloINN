#include "DelphesRootLoader.h"

using namespace MultijetExtractor;

DelphesRootLoader::DelphesRootLoader(const char* file)
        : chain("Delphes"), current_entry(0) {
    TString input_file(file);
    chain.Add(input_file);

    tree_reader = new ExRootTreeReader(&chain);
    n_entries = tree_reader->GetEntries();
    branch = tree_reader->UseBranch("EFlowMerger");
}

DelphesRootLoader::~DelphesRootLoader() {
    delete tree_reader;
}

bool DelphesRootLoader::next_event(std::vector<HepMC::FourVector>& constits) {
    if (current_entry >= n_entries) return false;

    tree_reader->ReadEntry(current_entry);
    constits.clear();

    for (int i = 0; i < branch->GetEntriesFast(); i++) {
        Tower* pflow = (Tower*)branch->At(i);
        TLorentzVector mom = pflow->P4();
        constits.push_back(HepMC::FourVector(mom.Px(), mom.Py(), mom.Pz(), mom.E()));
    }

    current_entry++;
    return true;
}
