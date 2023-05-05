#include "HepMcLoader.h"

#include "HepMC/GenEvent.h"

using namespace MultijetExtractor;

HepMcLoader::HepMcLoader(char const* file, bool _hard_process)
    : hard_process(_hard_process), hepmc_in(file, std::ios::in) {

}

bool HepMcLoader::next_event(std::vector<HepMC::FourVector>& constits) {
    HepMC::GenEvent* event = hepmc_in.read_next_event();
    if (!event) return false;
    constits.clear();

    if (hard_process) {
        HepMC::GenVertex* signal = event->signal_process_vertex();
        for (HepMC::GenVertex::particles_out_const_iterator p =
                signal->particles_out_const_begin();
                p != signal->particles_out_const_end(); ++p ) {
            constits.push_back((*p)->momentum());
        }
    } else {
        for (HepMC::GenEvent::particle_iterator p = event->particles_begin();
                p != event->particles_end(); ++p ) {
            if (!(*p)->end_vertex() && (*p)->status() == 1) {
                constits.push_back((*p)->momentum());
            }
        }
    }

    delete event;
    return true;
}
