#include <iostream>
#include <Eigen/Dense>

#include "hdf5.h"


#ifdef USE_MPI
#include <mpi.h>
#endif


int main(int argc, char **argv) {

#ifdef USE_MPI
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    int global_rows, global_cols, rows, cols;

    if (argc >= 3) {
        global_rows = std::stoi(argv[1]);
        global_cols = std::stoi(argv[2]);
    } else if (argc == 2) {
        global_rows = std::stoi(argv[1]);
        global_cols = global_rows;
    } else {
        throw std::invalid_argument("Please supply the dimensions of the matrix:\n\tOne argument for an n x n square matrix\n\tTwo arguments for an n x m rectangular matrix");
    }

#ifdef USE_MPI
    cols = global_cols / world_size + (global_cols % world_size >= rank ? 1 : 0);
    rows = global_rows;
#else
    rows = global_rows;
    cols = global_cols;
#endif

    Eigen::ArrayXXd matrix = Eigen::ArrayXXd::Random(rows, cols);

#ifdef USE_MPI
    auto file_access_template = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(file_access_template, MPI_COMM_WORLD, MPI_INFO_NULL);
#else
    auto file_access_template = H5P_DEFAULT;
#endif

    hid_t file_id = H5Fcreate("matrix.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, file_access_template);
    H5Pclose(file_access_template);

    hsize_t dims[2] = {(hsize_t) global_rows, (hsize_t) global_cols};
    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);

    hid_t dataset = H5Dcreate2(file_id, "matrix", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

#ifdef USE_MPI
    hsize_t start[2] = {0, (hsize_t) (rank * cols + (global_cols % world_size < rank ? global_cols % world_size : 0))};
#else
    hsize_t start[2] = {0, 0};
#endif

    hsize_t count[2] = {(hsize_t) rows, (hsize_t) cols};
    hsize_t stride[2] = {1, 1};

    hid_t file_dataspace = H5Dget_space(dataset);
    H5Sselect_hyperslab(file_dataspace, H5S_SELECT_SET, start, stride, count, nullptr);

    hid_t mem_dataspace = H5Screate_simple(2, count, nullptr);

    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, mem_dataspace, file_dataspace, H5P_DEFAULT, matrix.data());

    H5Sclose(file_dataspace);
    H5Dclose(dataset);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
