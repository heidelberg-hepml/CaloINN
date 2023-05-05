#include "NumpyFile.h"

#include <sstream>

using namespace MultijetExtractor;

const NumpyDtype NumpyDtype::FLOAT32 { "<f4", 4 };
const NumpyDtype NumpyDtype::INT64 { "<i8", 8 };

const char* header = "\x93NUMPY\x01\x00\x76\x00";
const size_t header_len = 10;

NumpyFile::NumpyFile(std::string fname, NumpyDtype _dtype, int _col_count)
    : col_count(_col_count), row_count(0), dtype(_dtype),
      file(fname, std::ios::out | std::ios::binary) {
    file.write(header, header_len);
    for (int i = 0; i < 127 - header_len; i++) {
        file.put(' ');
    }
    file.put('\n');
}

NumpyFile::~NumpyFile() {
    file.seekp(header_len);
    std::ostringstream oss;
    oss << "{'descr':'" << dtype.descr << "','fortran_order':False,'shape':("
        << row_count << "," << col_count << ")}";
    file << oss.str();
}

void NumpyFile::write_row(char* data) {
    file.write(data, col_count * dtype.size);
    row_count++;
}
