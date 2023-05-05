#ifndef INCLUDE_GUARD_B0213270_A1FA_436B_875E_988B5F18E430
#define INCLUDE_GUARD_B0213270_A1FA_436B_875E_988B5F18E430

#include "HepMC/SimpleVector.h"
#include <vector>

namespace MultijetExtractor {

void sort_by_pt(std::vector<HepMC::FourVector>& constits);
void momenta_to_buffer(std::vector<HepMC::FourVector>& constits, float* out_buf,
                       int max_length=-1);

}

#endif //INCLUDE_GUARD_B0213270_A1FA_436B_875E_988B5F18E430
