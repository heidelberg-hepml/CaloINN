#ifndef INCLUDE_GUARD_CD0787A9_1DAB_4262_9C61_3506AE798D2E
#define INCLUDE_GUARD_CD0787A9_1DAB_4262_9C61_3506AE798D2E
#include <string>
#include <fstream>

namespace MultijetExtractor {

struct NumpyDtype {
    char const* descr;
    size_t size;

    static const NumpyDtype FLOAT32;
    static const NumpyDtype INT64;
};

class NumpyFile {
private:
    int col_count, row_count;
    NumpyDtype dtype;
    std::ofstream file;
public:
    NumpyFile(std::string fname, NumpyDtype dtype, int _col_count);
    ~NumpyFile();
    void write_row(char* data);
};

}

#endif //INCLUDE_GUARD_CD0787A9_1DAB_4262_9C61_3506AE798D2E
