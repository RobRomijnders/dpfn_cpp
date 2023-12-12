#ifndef SRC_BP_HPP_
#define SRC_BP_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <list>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

namespace py = pybind11;

#include "./util.hpp"


std::vector<float> make_obs_messages(
    const int& num_users,
    const int& num_time_steps,
    const float& alpha,
    const float& beta,
    const py::array_t<int>& observations) {
  // Makes the observation messages
  //
  // Args:
  //   num_users: Number of users
  //   num_time_steps: Number of time steps
  //   alpha: False negative rate
  //   beta: False positive rate
  //   observations: Numpy array of observations
  //
  // Returns:
  //   obs_messages: Vector of observation messages,
  //     length num_users*num_time_steps*4

  float obs_distro[2][4] {
    {1-beta, 1-beta, alpha, 1-beta}, {beta, beta, 1-alpha, beta}};

  std::vector<float> obs_messages(num_users*num_time_steps*4, 1.f);

  auto obs_array = observations.unchecked<2>();
  int num_observations = obs_array.shape(0);

  int idx, user, timestep, outcome;
  for (int num_obs = 0; num_obs < num_observations; num_obs++) {
    // Each observation is a tuple of (user, timestep, outcome)
    user = obs_array(num_obs, 0);
    timestep = obs_array(num_obs, 1);
    outcome = obs_array(num_obs, 2);

    idx = user*num_time_steps*4 + timestep*4;
    obs_messages[idx] *= obs_distro[outcome][0];
    obs_messages[idx+1] *= obs_distro[outcome][1];
    obs_messages[idx+2] *= obs_distro[outcome][2];
    obs_messages[idx+3] *= obs_distro[outcome][3];
  }

  return obs_messages;
}

int calc_transition_probs(
    const int& num_user,
    const int& num_time_steps,
    const float& probab0,
    const float& probab1,
    const float& rho_rdp,
    const float& a_rdp,
    const float& clip_lower,
    const float& clip_upper,
    const int * past_contacts,
    const std::vector<std::vector<float>>& p_infected,
    float * transition_probs) {
  // Calculates the transition probabilities for a single user
  //
  // Args:
  //   num_user: User ID that we'll update
  //   num_time_steps: Number of time steps
  //   probab0: Probability of spontaneous infection
  //   probab1: Probability of infection from contact
  //   rho_rdp: RDP privacy parameter
  //   a_rdp: RDP privacy parameter
  //   clip_lower: Lower clipping value
  //   clip_upper: Upper clipping value
  //   past_contacts: Array of past contacts, of length num_users*CTC_SIZE*2
  //   p_infected: Array of p_infected, of length num_users*num_time_steps
  //   transition_probs: Array of transition probabilities to write results

  for (int t = 0; t < num_time_steps; t++) {
    transition_probs[t] = std::log(1.f-probab0);}

  int timestep = 0;
  int sender = -1;
  int idx = 0;

  int num_contacts[num_time_steps] = {0};

  for (int num_contact = 0; num_contact < CTC_SIZE; num_contact++) {
    idx = num_user*CTC_SIZE*2 + num_contact*2;
    timestep = past_contacts[idx];
    sender = past_contacts[idx+1];

    if (timestep < 0)
        break;

    transition_probs[timestep] += std::log(
      1.-probab1*p_infected[sender][timestep]);
    num_contacts[timestep]++;
  }

  if (rho_rdp > 0.f) {
    assert(a_rdp > 1.f);
    // Random number generator
    // TODO(rob): connect to global random seed or something
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Calculate sensitivity
    // Pull clipping values from defaults numbers back to [0, 1]
    float c_upper = std::min(clip_upper, 0.99999f);
    float c_lower = std::max(clip_lower, 0.00001f);

    // Equation as derived in 'Protect your score' paper
    float sensitivity =  std::abs(
      std::log(1-c_upper*probab1) - std::log(1-c_lower*probab1));


    float sigma_squared, sigma;  // Gaussian noise variance
    int num_contacts_abs;
    float float_upper, float_lower;  // Clipping limits that are public

    for (int t = 0; t < num_time_steps; t++) {
      if (num_contacts[t] > 0) {
        num_contacts_abs = num_contacts[t];

        sigma_squared = std::pow(sensitivity, 2) * a_rdp / (
          2 * rho_rdp);
        sigma = std::sqrt(sigma_squared);

        transition_probs[t] += (
          dist(generator) * sigma - 0.5 * sigma_squared);

        // Public knowledge of clipping limits
        float_upper = (std::log(1.f-probab0)
                      + num_contacts_abs*std::log(1. - probab1*c_lower));
        float_lower = (std::log(1.f-probab0)
                      + num_contacts_abs*std::log(1. - probab1*c_upper));

        transition_probs[t] = std::max(
          float_lower, std::min(float_upper, transition_probs[t]));
      }
    }
  }

  // Convert to probabilities
  for (int t = 0; t < num_time_steps; t++) {
    transition_probs[t] = std::exp(transition_probs[t]);
  }

  return 1;
  }


int forward_backward_user(
    const int& num_user,
    const int& num_time_steps,
    const float& probab0,
    const float& probab1,
    const float& g_param,
    const float& h_param,
    const float& rho_rdp,
    const float& a_rdp,
    const float& clip_lower,
    const float& clip_upper,
    const int * past_contacts,
    const std::vector<std::vector<float>>& p_infected,
    const std::vector<float>& obs_messages,
    array3<float>& post_exp) {
  // Forward-backward algorithm for a single user
  float smallnum = 1E-9;

  float transition_probs[num_time_steps];

  calc_transition_probs(
    num_user, num_time_steps, probab0, probab1, rho_rdp, a_rdp, clip_lower,
    clip_upper, past_contacts, p_infected, transition_probs);

  float forward_messages[num_time_steps*4];
  float backward_messages[num_time_steps*4];
  std::vector<float> (num_time_steps*4, 1.f);

  // Forward pass
  forward_messages[0] = 1.-probab0;
  forward_messages[1] = probab0;
  forward_messages[2] = smallnum;
  forward_messages[3] = smallnum;

  float varmessage0, varmessage1, varmessage2, varmessage3;

  for (int t = 1; t < num_time_steps; t++) {
    varmessage0 = (
      obs_messages[num_user*num_time_steps*4 + (t-1)*4] *
      forward_messages[(t-1)*4]);
    forward_messages[t*4] = (varmessage0 * transition_probs[t-1]);

    varmessage1 = (
      obs_messages[num_user*num_time_steps*4 + (t-1)*4+1] *
      forward_messages[(t-1)*4+1]);
    forward_messages[t*4+1] = (
      varmessage0 * (1.f - transition_probs[t-1]) +
      varmessage1 * (1.f - g_param));

    varmessage2 = (
      obs_messages[num_user*num_time_steps*4 + (t-1)*4+2] *
      forward_messages[(t-1)*4+2]);
    forward_messages[t*4+2] = (
      varmessage1 * g_param +
      varmessage2 * (1-h_param));

    varmessage3 = (
      obs_messages[num_user*num_time_steps*4 + (t-1)*4+3] *
      forward_messages[(t-1)*4+3]);
    forward_messages[t*4+3] = (
      varmessage2 * h_param +
      varmessage3);
  }

  // Backward pass
  backward_messages[(num_time_steps-1)*4] = 1.f;
  backward_messages[(num_time_steps-1)*4+1] = 1.f;
  backward_messages[(num_time_steps-1)*4+2] = 1.f;
  backward_messages[(num_time_steps-1)*4+3] = 1.f;

  for (int t = num_time_steps - 2; t >= 0; t--) {
    varmessage0 = (
      obs_messages[num_user*num_time_steps*4 + (t+1)*4] *
      backward_messages[(t+1)*4]);
    varmessage1 = (
      obs_messages[num_user*num_time_steps*4 + (t+1)*4+1] *
      backward_messages[(t+1)*4+1]);
    varmessage2 = (
      obs_messages[num_user*num_time_steps*4 + (t+1)*4+2] *
      backward_messages[(t+1)*4+2]);
    varmessage3 = (
      obs_messages[num_user*num_time_steps*4 + (t+1)*4+3] *
      backward_messages[(t+1)*4+3]);

    backward_messages[t*4] = (
      varmessage0 * transition_probs[t]+
      varmessage1 * (1.f - transition_probs[t]));

    backward_messages[t*4+1] = (
      varmessage1 * (1-g_param) +
      varmessage2 * g_param);

    backward_messages[t*4+2] = (
      varmessage2 * (1-h_param) +
      varmessage3 * h_param);

    backward_messages[t*4+3] = (
      varmessage3);
  }

  // Compute posterior expectations
  for (int t = 0; t < num_time_steps; t++) {
    float rowsum = 0.f;
    for (int state = 0; state < 4; state++) {
      post_exp[num_user][t][state] = (
        forward_messages[t*4 + state] *
        backward_messages[t*4 + state] *
        obs_messages[num_user*num_time_steps*4 + t*4 + state]) + smallnum;
      rowsum += post_exp[num_user][t][state];
    }
    // Normalise to 1
    for (int state = 0; state < 4; state++) {
      post_exp[num_user][t][state] /= rowsum;
    }
  }

  return 1;
}


py::array_t<float> forward_backward_user_wrapped(
  const int& num_user,
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
  const array2<float>& p_infected,
  const py::array_t<int>& observations,
  const py::array_t<int>& contacts) {
  // Wraps the forward_backward_user for use from Python
  //
  // Use this function only for unit-testing.
  // This function does a copy which is slow

  std::vector<float> obs_messages = make_obs_messages(
    num_users, num_time_steps, alpha, beta, observations);

  // Make post_exp a 3d array
  std::vector<std::vector<std::vector<float>>> post_exp(
    num_users, std::vector<std::vector<float>>(
      num_time_steps, std::vector<float>(4, 0.0)));
  std::vector<std::vector<std::vector<float>>>& post_exp_ref {post_exp};

  // Move past contacts to array
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  forward_backward_user(
    num_user, num_time_steps, probab0, probab1, g_param, h_param, rho_rdp,
    a_rdp, clip_lower, clip_upper, past_contacts, p_infected,
    obs_messages, post_exp_ref);

  delete[] past_contacts;
  return py::array(py::cast(post_exp));
  }

array3<float> bp_full(
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
  const py::array_t<int>& contacts
) {
  // We will not modify any python objects, so we can release the GIL
  std::chrono::time_point<std::chrono::system_clock> m_StartTime =
    std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds;

  py::gil_scoped_release release;

  // Initialize p_inf
  std::vector<std::vector<float>> p_inf(
    num_users, std::vector<float>(num_time_steps, 0.0));
  std::vector<std::vector<float>>& p_inf_ref {p_inf};

  // Make obs messages
  std::vector<float> obs_messages = make_obs_messages(
    num_users, num_time_steps, alpha, beta, observations);

  // Make post_exp a 3d array
  std::vector<std::vector<std::vector<float>>> post_exp(
    num_users, std::vector<std::vector<float>>(
      num_time_steps, std::vector<float>(4, 0.0)));
  std::vector<std::vector<std::vector<float>>>& post_exp_ref {post_exp};

  // Move past contacts to array
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  // Split users among cores
  std::vector<int> data_array = divide_work(num_users, num_workers);
  std::vector<int> user_id(num_workers+1, 0);
  for (int i = 1; i < num_workers+1; i++) {
    user_id[i] = user_id[i-1] + data_array[i-1];
  }

  elapsed_seconds = std::chrono::system_clock::now() - m_StartTime;
  std::cout << "Preamble time: " << elapsed_seconds.count() << "s\n";

  float message = 0.0;
  float rho_rdp_set = -1.;
  float a_rdp_set = -1.;
  float clip_lower_set = -1.;
  float clip_upper_set = 10.;

  for (int num_round = 0; num_round < num_rounds; num_round++) {
    if (num_round == num_rounds - 1) {
      // Only run DP on last round
      rho_rdp_set = rho_rdp;
      a_rdp_set = a_rdp;
      clip_lower_set = clip_lower;
      clip_upper_set = clip_upper;
    }

    // Update p_inf
    for (int n_user = 0; n_user < num_users; n_user++) {
      for (int t = 0; t < num_time_steps; t++) {
        message = post_exp[n_user][t][2];

        if (quantization > 0) {
          message = std::floor(message * quantization) / quantization;}

        message = std::min(clip_upper_set, message);
        message = std::max(clip_lower_set, message);

        p_inf[n_user][t] = message;
      }
    }

    // Divide the updates per user among the workers
    std::thread tlist[num_workers];
    for (int num_worker = 0; num_worker < num_workers; ++num_worker) {
      tlist[num_worker] = std::thread([&, num_worker]() {
        int user_start = user_id[num_worker];
        int user_end = user_id[num_worker+1];

        for (int user = user_start; user < user_end; user++) {
          forward_backward_user(
            user, num_time_steps, probab0, probab1, g_param, h_param,
            rho_rdp_set, a_rdp_set, clip_lower_set, clip_upper_set,
            past_contacts, p_inf_ref, obs_messages, post_exp_ref);
        }
        });
      }

    for (int num_worker = 0; num_worker < num_workers; ++num_worker) {
      tlist[num_worker].join();
    }
  }

  delete[] past_contacts;
  return post_exp;
}

#endif  // SRC_BP_HPP_
