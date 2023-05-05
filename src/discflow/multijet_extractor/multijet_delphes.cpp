#include "DelphesRootLoader.h"
#include "FastjetFinder.h"
#include "NumpyFile.h"
#include "Util.h"

#include <cstdlib>
#include <iostream>

using namespace MultijetExtractor;

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " input_file output_file "
                  << "min_jet_count max_jet_count jet_pt_min jet_r" << std::endl;
        return 1;
    }
    int min_jet_count = std::atoi(argv[3]);
    int max_jet_count = std::atoi(argv[4]);
    double jet_pt_min = std::atof(argv[5]);
    double jet_r      = std::atof(argv[6]);
    int n_columns = 4*max_jet_count;

    DelphesRootLoader loader(argv[1]);
    NumpyFile np_file(argv[2], NumpyDtype::FLOAT32, n_columns);
    FastjetFinder fastjet(jet_r, jet_pt_min);
    std::vector<HepMC::FourVector> constits;
    float* out_buf = new float[n_columns];
    for (;;) {
        if (!loader.next_event(constits)) break;
        fastjet.process_event(constits);
        if (constits.size() < min_jet_count || constits.size() > max_jet_count) continue;
        momenta_to_buffer(constits, out_buf, max_jet_count);
        np_file.write_row((char*)out_buf);
    }
    delete out_buf;
    return 0;
}
