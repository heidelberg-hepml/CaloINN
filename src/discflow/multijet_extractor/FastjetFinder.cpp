#include "FastjetFinder.h"

#include "fastjet/ClusterSequence.hh"

using namespace MultijetExtractor;

FastjetFinder::FastjetFinder(double r, double pt_min)
    : jet_def(fastjet::antikt_algorithm, r),
      selector(fastjet::SelectorPtMin(pt_min)) {}

void FastjetFinder::process_event(std::vector<HepMC::FourVector>& constits) {
    std::vector<fastjet::PseudoJet> towers;
    for (HepMC::FourVector fv : constits) {
        towers.push_back(fastjet::PseudoJet(fv.px(), fv.py(), fv.pz(), fv.e()));
    }
    fastjet::ClusterSequence cluster(towers, jet_def);
    std::vector<fastjet::PseudoJet> antikt_jets
        = fastjet::sorted_by_pt(selector(cluster.inclusive_jets()));

    constits.clear();
    for (fastjet::PseudoJet mom : antikt_jets) {
        constits.push_back(HepMC::FourVector(mom.px(), mom.py(), mom.pz(), mom.e()));
    }
}
