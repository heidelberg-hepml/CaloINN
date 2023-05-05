#include "Util.h"

using namespace MultijetExtractor;

void MultijetExtractor::sort_by_pt(std::vector<HepMC::FourVector>& constits) {
    std::sort(constits.begin(), constits.end(),
        [ ](const HepMC::FourVector& lhs, const HepMC::FourVector& rhs) {
            return lhs.perp2() > rhs.perp2();
        }
    );
}

void MultijetExtractor::momenta_to_buffer(std::vector<HepMC::FourVector>& constits,
                                          float* out_buf, int max_length) {
    if (max_length == -1) max_length = constits.size();
    int i = 0;
    for (; i < constits.size() && i < max_length; i++) {
        out_buf[4*i + 0] = constits[i].e();
        out_buf[4*i + 1] = constits[i].px();
        out_buf[4*i + 2] = constits[i].py();
        out_buf[4*i + 3] = constits[i].pz();
    }
    for (; i < max_length; i++) {
        out_buf[4*i + 0] = 0.;
        out_buf[4*i + 1] = 0.;
        out_buf[4*i + 2] = 0.;
        out_buf[4*i + 3] = 0.;
    }
}

