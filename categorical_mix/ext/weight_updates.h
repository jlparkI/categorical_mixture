#ifndef WEIGHTED_CATMIX_COUNT_CALCS_H
#define WEIGHTED_CATMIX_COUNT_CALCS_H

#include <stdint.h>
#include <stdlib.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

void getWeightedCountCExt(py::array_t<uint8_t, py::array::c_style> x,
        py::array_t<double, py::array::c_style> wcount,
        py::array_t<double, py::array::c_style> resp,
        int n_threads);

void *getWeightedCountCExt_worker(uint8_t *x, double *resp,
        double *wcount, int wcount_dim1, int wcount_dim2,
        int x_dim0, int x_dim1, int startRow, int endRow);


int wcountSafetyChecks(py::array_t<uint8_t, py::array::c_style> x,
        py::array_t<double, py::array::c_style> wcount,
        py::array_t<double, py::array::c_style> resp,
        int n_threads);


#endif
