/* Contains the wrapper code for the C++ extension for alignment
 * calculations.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>    // Enables automatic type conversion for C++, python containers
#include <string>
#include "responsibility_calcs.h"
#include "weight_updates.h"
#include "catmix_utilities.h"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(categorical_mix_cpp_ext, m){
    m.def("getProbsCExt", &getProbsCExt, py::call_guard<py::gil_scoped_release>());
    m.def("mask_terminal_deletions", &mask_terminal_deletions, py::call_guard<py::gil_scoped_release>());
    m.def("getProbsCExt_masked", &getProbsCExt_masked, py::call_guard<py::gil_scoped_release>());
    m.def("getWeightedCountCExt", &getWeightedCountCExt, py::call_guard<py::gil_scoped_release>());
}
