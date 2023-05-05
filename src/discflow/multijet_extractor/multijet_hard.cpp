#include "HepMcLoader.h"
#include "NumpyFile.h"
#include "Util.h"

#include <cstdlib>
#include <iostream>

using namespace MultijetExtractor;

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " input_file output_file "
                  << "min_jet_count max_jet_count" << std::endl;
        return 1;
    }
    int min_jet_count = std::atoi(argv[3]);
    int max_jet_count = std::atoi(argv[4]);
    int n_columns = 4*max_jet_count;

    HepMcLoader loader(argv[1], true);
    NumpyFile np_file(argv[2], NumpyDtype::FLOAT32, n_columns);
    std::vector<HepMC::FourVector> constits;
    float* out_buf = new float[n_columns];
    for (;;) {
        if (!loader.next_event(constits)) break;
        if (constits.size() < min_jet_count || constits.size() > max_jet_count) continue;
        sort_by_pt(constits);
        momenta_to_buffer(constits, out_buf, max_jet_count);
        np_file.write_row((char*)out_buf);
    }
    delete out_buf;
    return 0;
}
