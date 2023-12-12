#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <iostream>
#include <iomanip>
#include <list>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define CTC_SIZE 900
#define NUM_DUMP_FEATURES 4

namespace py = pybind11;

// Shortname for array types
template<typename T>
using array1 = std::vector<T>;

template<typename T>
using array2 = std::vector<array1<T>>;

template<typename T>
using array3 = std::vector<array2<T>>;

#include "./fn.hpp"
#include "./bp.hpp"
#include "./util.hpp"

PYBIND11_MODULE(dpfn_util, m) {
  m.doc() = "pybind11 dpfn_util plugin";

  m.def("numpy_change", &numpy_change, py::arg().noconvert(), R"pbdoc(
    Change a numpy array )pbdoc");

  m.def("divide_work",
    [] (int num_jobs, int num_workers) {
      return py::array(py::cast(divide_work(num_jobs, num_workers)));},
    py::kw_only(),
    py::arg("num_jobs").noconvert(),
    py::arg("num_workers").noconvert(),
    "Spreads the work among users.");

  m.def("contact_flip",
    [] (int num_users, py::array_t<int>& contacts) {
      int * past_contacts = new int[num_users*CTC_SIZE*2];
      contact_flip(num_users, contacts, past_contacts);
      return py::array(py::buffer_info(
        past_contacts,  /* Pointer to buffer */
        sizeof(int),  /* Size of one scalar */
        py::format_descriptor<int>::format(),
        1,  /* Number of dimensions */
        {num_users*CTC_SIZE*2},  /* Buffer dimensions */
        {sizeof(int)})); /* Strides (in bytes) for each index */
      },
    py::kw_only(),
    py::arg("num_users").noconvert(),
    py::arg("contacts").noconvert(),
    py::return_value_policy::take_ownership,
    R"pbdoc( Sort contacts by receiver. )pbdoc");

  m.def("iter_sequences",
    [] (int num_time_steps) {
      return py::array(py::cast(iter_sequences(num_time_steps)));},
    py::kw_only(),
    py::arg("num_time_steps").noconvert(),
    py::return_value_policy::move,
    R"pbdoc( Iterate over possible sequences. )pbdoc");

  m.def("make_seq_array_hot",
    [] (array2<int> seq_array, int num_time_steps) {
      return py::array(py::cast(
        make_seq_array_hot(seq_array, num_time_steps)));},
    py::arg("seq_array"),
    py::kw_only(),
    py::arg("num_time_steps").noconvert(),
    py::return_value_policy::move,
    R"pbdoc( Makes the hot seq_array. )pbdoc");

  m.def("fn_full_func",
    [](
      const int& num_workers,
      const int& num_rounds,
      const int& num_users,
      const int& num_time_steps,
      const float& probab0,
      const float& probab1,
      const float& g_param,
      const float& h_param,
      const float& alpha,
      const float& beta,
      const float& rho_rdp,
      const float& a_rdp,
      const float& clip_lower,
      const float& clip_upper,
      const int& quantization,
      const py::array_t<int>& observations,
      const py::array_t<int>& contacts,
      const py::array_t<int>& users_age,
      const int dedup_contacts = 0){
        std::pair<array3<float>, std::vector<int>> result = fn_full(
          num_workers,
          num_rounds,
          num_users,
          num_time_steps,
          probab0,
          probab1,
          g_param,
          h_param,
          alpha,
          beta,
          rho_rdp,
          a_rdp,
          clip_lower,
          clip_upper,
          quantization,
          observations,
          contacts,
          users_age,
          dedup_contacts);
        // TODO(rob): get rid of the copy in the next line!
        // int a = 1;
        // return std::pair(py::array(py::cast(post_exp)), a);
        return std::pair<py::array_t<float>, py::array_t<int>>(
          py::array(py::cast(result.first)),
          py::array(py::cast(result.second)));
      },
    py::kw_only(),
    py::arg("num_workers").noconvert(),
    py::arg("num_rounds").noconvert(),
    py::arg("num_users").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("probab0").noconvert(),
    py::arg("probab1").noconvert(),
    py::arg("g_param").noconvert(),
    py::arg("h_param").noconvert(),
    py::arg("alpha"),
    py::arg("beta"),
    py::arg("rho_rdp"),
    py::arg("a_rdp"),
    py::arg("clip_lower"),
    py::arg("clip_upper"),
    py::arg("quantization").noconvert(),
    py::arg("observations").noconvert(),
    py::arg("contacts").noconvert(),
    py::arg("users_age").noconvert(),
    py::arg("dedup_contacts") = 0,
    py::return_value_policy::move,
    "Calculate multiple rounds of FN update");

m.def("fn_features_dump",
    [](
      const int& num_workers,
      const int& num_users,
      const int& num_time_steps,
      const py::array_t<float>& q_marginal,
      const py::array_t<int>& contacts,
      const py::array_t<int>& users_age){
        array3<int> datadump = fn_data_dump(
          num_workers,
          num_users,
          num_time_steps,
          q_marginal,  // post exp
          contacts,
          users_age);

        return py::array(py::cast(datadump));
      },
    py::kw_only(),
    py::arg("num_workers").noconvert(),
    py::arg("num_users").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("q_marginal").noconvert(),
    py::arg("contacts").noconvert(),
    py::arg("users_age").noconvert(),
    py::return_value_policy::move,
    "Prepare data dump for FN");

  m.def("fn_wrapped_func", &fn_wrapped,
    py::kw_only(),
    py::arg("user_start").noconvert(),
    py::arg("user_end").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("probab0").noconvert(),
    py::arg("probab1").noconvert(),
    py::arg("rho_rdp"),
    py::arg("a_rdp"),
    py::arg("clip_lower"),
    py::arg("clip_upper"),
    py::arg("seq_array_hot"),
    py::arg("log_c_z_u"),
    py::arg("log_a_start"),
    py::arg("p_infected_matrix"),
    py::arg("contacts"),
    py::return_value_policy::move,
    "Calculates a full fn update. Use this function for unit tests only!");

  m.def("d_terms_wrapped", &d_terms_wrapped,
    py::kw_only(),
    py::arg("num_user").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("q_marginal"),
    py::arg("contacts"),
    py::arg("p0").noconvert(),
    py::arg("p1").noconvert(),
    py::arg("rho_rdp"),
    py::arg("a_rdp"),
    py::arg("clip_lower"),
    py::arg("clip_upper"),
    py::arg("dedup_contacts") = 0,
    py::return_value_policy::move,
    "Precompute d penalties with plain cpp");

  m.def("precompute_a_terms",
    [] (int num_time_steps, float probab0, float g_param, float h_param,
        array2<int> sequences) {
      return py::array(py::cast(precompute_a_terms(
        num_time_steps, probab0, g_param, h_param, sequences)));},
    py::kw_only(),
    py::arg("num_time_steps").noconvert(),
    py::arg("probab0").noconvert(),
    py::arg("g_param").noconvert(),
    py::arg("h_param").noconvert(),
    py::arg("sequences"),
    py::return_value_policy::move,
    R"pbdoc( Precomputes the a_terms for FN. )pbdoc");

  m.def("precompute_c_terms",
    [] (int num_users, float alpha, float beta, py::array_t<int>& observations,
        array3<int> seq_hot) {
      return py::array(py::cast(precompute_c_terms(
        num_users, alpha, beta, observations, seq_hot)));},
    py::kw_only(),
    py::arg("num_users").noconvert(),
    py::arg("alpha").noconvert(),
    py::arg("beta").noconvert(),
    py::arg("observations").noconvert(),
    py::arg("seq_hot"),
    py::return_value_policy::move,
    R"pbdoc( Precomputes the c_terms for FN. )pbdoc");

  m.def("subtract", [](int i, int j) { return i - j; });

  // BP related functions
  m.def("make_obs_messages",
    [] (
      const int& num_users,
      const int& num_time_steps,
      const float& alpha,
      const float& beta,
      const py::array_t<int>& observations) {
      return py::array(py::cast(make_obs_messages(
        num_users, num_time_steps, alpha, beta, observations)));},
    py::kw_only(),
    py::arg("num_users").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("alpha").noconvert(),
    py::arg("beta").noconvert(),
    py::arg("observations"),
    py::return_value_policy::move,
    "Make observation messages array");

  m.def("forward_backward_user_wrapped", &forward_backward_user_wrapped,
    py::kw_only(),
    py::arg("num_user").noconvert(),
    py::arg("num_users").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("probab0").noconvert(),
    py::arg("probab1").noconvert(),
    py::arg("g_param").noconvert(),
    py::arg("h_param").noconvert(),
    py::arg("alpha").noconvert(),
    py::arg("beta").noconvert(),
    py::arg("rho_rdp"),
    py::arg("a_rdp"),
    py::arg("clip_lower"),
    py::arg("clip_upper"),
    py::arg("p_infected"),
    py::arg("observations"),
    py::arg("contacts"),
    py::return_value_policy::move,
    "Make observation messages array");

    m.def("bp_full_func",
    [](
      const int& num_workers,
      const int& num_rounds,
      const int& num_users,
      const int& num_time_steps,
      const float& probab0,
      const float& probab1,
      const float& g_param,
      const float& h_param,
      const float& alpha,
      const float& beta,
      const float& rho_rdp,
      const float& a_rdp,
      const float& clip_lower,
      const float& clip_upper,
      const int& quantization,
      const py::array_t<int>& observations,
      const py::array_t<int>& contacts){
        array3<float> post_exp = bp_full(
          num_workers,
          num_rounds,
          num_users,
          num_time_steps,
          probab0,
          probab1,
          g_param,
          h_param,
          alpha,
          beta,
          rho_rdp,
          a_rdp,
          clip_lower,
          clip_upper,
          quantization,
          observations,
          contacts);
        // TODO(rob): get rid of the copy in the next line!
        return py::array(py::cast(post_exp));
      },
    py::kw_only(),
    py::arg("num_workers").noconvert(),
    py::arg("num_rounds").noconvert(),
    py::arg("num_users").noconvert(),
    py::arg("num_time_steps").noconvert(),
    py::arg("probab0").noconvert(),
    py::arg("probab1").noconvert(),
    py::arg("g_param").noconvert(),
    py::arg("h_param").noconvert(),
    py::arg("alpha"),
    py::arg("beta"),
    py::arg("rho_rdp"),
    py::arg("a_rdp"),
    py::arg("clip_lower"),
    py::arg("clip_upper"),
    py::arg("quantization").noconvert(),
    py::arg("observations").noconvert(),
    py::arg("contacts").noconvert(),
    py::return_value_policy::move,
    "Calculate multiple rounds of BP update");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
