#ifndef SRC_FN_HPP_
#define SRC_FN_HPP_

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
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

#include "./util.hpp"

struct Contact {
  int sender;
  int timestep;
  float pinf;
};


int numpy_change(
    py::array_t<float, py::array::c_style | py::array::forcecast> &array) {
  auto r = array.mutable_unchecked<2>();
  r(0, 0) = 13.0;

  // Divide total number over range
  std::thread tlist[8];
  for (int i = 0; i < 8; ++i) {
    tlist[i] = std::thread([&, i]() {
      std::cout << "hello " << i << std::endl; });
  }

  for (int i = 0; i < 8; ++i) {
    tlist[i].join();
  }
  return 0;
}


std::vector<int> divide_work(
    const int& num_jobs,
    const int& num_workers) {
  // Constructs a vector with the amount of jobs for each worker
  //
  // Args:
  //   num_jobs: number of jobs to be divided
  //   num_workers: number of workers
  //
  // Returns:
  //   data_array: vector of length num_workers with the
  //       amount of jobs for each worker
  std::vector<int> data_array(num_workers, 0);

  int base = num_jobs / num_workers;
  int num_add_one = num_jobs % num_workers;

  for (int i = 0; i < num_workers; i++) {
    data_array[i] = base;
    if (i < num_add_one)
      data_array[i] += 1;
  }

  return data_array;
}


int add_lognormal_noise_rdp(
      float * d_terms,
      const int& num_time_steps,
      const float& rho_rdp,
      const float& a_rdp,
      const float& sensitivity) {
  // Adds lognormal noise to the d_terms
  assert(sensitivity > 1E-5);
  assert(a_rdp > 1.0);

  // Random number generator
  // TODO(rob): connect to global random seed or something
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::normal_distribution<double> dist(0.0, 1.0);

  float sigma_squared_lognormal = std::pow(sensitivity, 2) * a_rdp / (
    2 * rho_rdp);
  float sigma_lognormal = std::sqrt(sigma_squared_lognormal);

  for (int timestep; timestep < num_time_steps; timestep++) {
    d_terms[timestep] += (
      sigma_lognormal * dist(generator) - 0.5 * sigma_squared_lognormal);
  }

  return 1;
  }

int run_contacts_dterms_repeats(
    const int& num_user,
    const float& probab1,
    const array2<float>& q_marginal,
    const int * past_contacts,
    int * happened,
    float * d_terms_array
) {
  // Remove repeating contacts!
  //
  // Some differential privacy methods are based on the assumption that each
  // sender sends only one message. In the case of repeating contacts in the
  // time window, we only take the last contact.
  int num_contacts = CTC_SIZE;
  int num_contacts_nonzero = 0;

  int idx, sender, timestep;
  float pinf;

  // This unordered map will keep track of all senders to this user in this time
  // window. The map will keep only the contact with the highest timestep.
  std::unordered_map<int, Contact> contacts_seen;

  for (int num_contact = 0; num_contact < num_contacts; num_contact++) {
      idx = num_user*CTC_SIZE*2 + num_contact*2;
      timestep = past_contacts[idx];
      if (timestep < 0)
          break;

      sender = past_contacts[idx+1];
      pinf = q_marginal[sender][timestep];

      // Check if sender has been seen already
      if (contacts_seen.count(sender) > 0) {
        Contact other = contacts_seen[sender];
        if (other.timestep < timestep) {
          // Replace in dterms
          d_terms_array[other.timestep+1] -= std::log(1. - probab1*other.pinf);
          d_terms_array[timestep+1] += std::log(1. - probab1*pinf);

          // Replace in happened
          happened[other.timestep + 1] -= 1;
          happened[timestep + 1] += 1;

          // Replace in dict
          contacts_seen[sender].timestep = timestep;
          contacts_seen[sender].pinf = pinf;
        }
      } else {
        Contact contact_add = {sender, timestep, pinf};
        contacts_seen[sender] = contact_add;

        happened[timestep + 1] += 1;
        d_terms_array[timestep+1] += std::log(1. - probab1*pinf);

        num_contacts_nonzero += 1;
      }
    }
  return num_contacts_nonzero;
}

int run_contacts_dterms(
    const int& num_user,
    const float& probab1,
    const array2<float>& q_marginal,
    const int * past_contacts,
    int * happened,
    float * d_terms_array
) {
  int num_contacts = CTC_SIZE;
  int num_contacts_nonzero = 0;

  int idx, sender, timestep;
  float pinf;

  for (int num_contact = 0; num_contact < num_contacts; num_contact++) {
      idx = num_user*CTC_SIZE*2 + num_contact*2;
      timestep = past_contacts[idx];
      if (timestep < 0)
          break;

      happened[timestep + 1] += 1;

      sender = past_contacts[idx+1];
      pinf = q_marginal[sender][timestep];
      d_terms_array[timestep+1] += std::log(1. - probab1*pinf);

      num_contacts_nonzero += 1;
    }
  return num_contacts_nonzero;
}


array2<float> precompute_d_terms(
  const int& num_user,
  const int& num_time_steps,
  const array2<float>& q_marginal,
  const int * past_contacts,
  const float& probab0,
  const float& probab1,
  const float& rho_rdp,
  const float& a_rdp,
  const float& clip_lower,
  const float& clip_upper,
  const int dedup_contacts = 0) {
  // Precomputes the d_terms
  //
  // Args:
  //   num_user: user index
  //   num_time_steps: number of time steps
  //   q_marginal: current estimate of covidscore
  //   past_contacts: past contacts, array in [num_users, num_contacts, 2]
  //   probab0: probability of starting in state 0
  //   probab1: probability transit S->E for an infected contact
  //   rho_rdp: rho privacy parameter for RDP
  //   a_rdp: a privacy parameter for RDP
  //   clip_lower: lower bound for clipping
  //   clip_upper: upper bound for clipping
  //
  // Returns:
  //   d_terms: 2d array with d_terms, first row is d_term, second row is
  //       d_no_term

  assert(past_contacts[(CTC_SIZE-1)*2] < 0);

  // Initialize data arrays
  float dterms_array[num_time_steps + 1] = {0.};
  int happened[num_time_steps + 1] = {0};

  int num_contacts_nonzero;

  if (dedup_contacts == 0) {
    // No deduplication
    num_contacts_nonzero = run_contacts_dterms(
      num_user,
      probab1,
      q_marginal,
      past_contacts,
      happened,
      dterms_array);
  } else if (dedup_contacts == 1) {
    // Deduplication, keep only last contact
    num_contacts_nonzero = run_contacts_dterms_repeats(
      num_user,
      probab1,
      q_marginal,
      past_contacts,
      happened,
      dterms_array);
  } else {
    throw std::invalid_argument("dedup_contacts should be 0 or 1");
  }

  if (a_rdp > 1) {
    // Pull clipping values from outrageous numbers back to default
    float c_upper = std::min(clip_upper, 0.99999f);
    float c_lower = std::max(clip_lower, 0.00001f);

    // For 0 contacts the d_no_term will be zero by public knowledge
    num_contacts_nonzero = std::max(num_contacts_nonzero, 1);

    // Equation as derived in 'Protect your score' paper
    float sensitivity =  std::abs(
      std::log(1-c_upper*probab1) - std::log(1-c_lower*probab1));

    add_lognormal_noise_rdp(
      dterms_array, num_time_steps, rho_rdp, a_rdp, sensitivity);

    float float_upper = num_contacts_nonzero*std::log(1. - probab1*c_lower);
    float float_lower = num_contacts_nonzero*std::log(1. - probab1*c_upper);

    for (int t = 0; t < num_time_steps+1; t++) {
      dterms_array[t] = std::max(
        float_lower, std::min(float_upper, dterms_array[t]));
    }
  }

  std::vector<std::vector<float>> d_terms(
    2, std::vector<float>(num_time_steps+1, 0.0f));

  float conv = 0.0;
  for (int t = 0; t < num_time_steps+1; t++) {
    if (happened[t] > 0) {
      // Only update if any contact happened. Otherwise, leave at default 0.0f
      conv = std::log(
        1 - (1-probab0)*std::exp(dterms_array[t])) - std::log(probab0);
      d_terms[0][t] = conv;

      // Also zero out d_no_term if nothing happened, this is public knowledge
      d_terms[1][t] = dterms_array[t];
    }
  }

  return d_terms;
}


array2<float> d_terms_wrapped(
  const int& num_user,
  const int& num_time_steps,
  const array2<float>& q_marginal,
  const py::array_t<int>& contacts,
  const float& probab0,
  const float& probab1,
  const float& rho_rdp,
  const float& a_rdp,
  const float& clip_lower,
  const float& clip_upper,
  const int dedup_contacts) {
  // Wraps the d_terms for use from Python
  int num_users = q_marginal.size();

  // Move past contacts to array
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  std::vector<std::vector<float>> d_terms =  precompute_d_terms(
    num_user,
    num_time_steps,
    q_marginal,
    past_contacts,
    probab0,
    probab1,
    rho_rdp,
    a_rdp,
    clip_lower,
    clip_upper,
    dedup_contacts);

  delete[] past_contacts;
  return d_terms;
  }


std::vector<float> precompute_a_terms(
  const int& num_time_steps,
  const float& probab0,
  const float& g_param,
  const float& h_param,
  const array2<int>& sequences) {
  // Precomputes the a_terms for FN.
  // TODO(rob): drop the possibility to start in state I or R
  //
  // Args:
  //   num_time_steps: number of time steps
  //   probab0: probability of starting in state 0
  //   g_param: probability of transition E->I
  //   h_param: probability of transition I->R
  //   sequences: 2d-array of sequences, in [num_contacts, 3]
  //
  // Returns:
  //   a_terms: vector of a_terms, of length [num_sequences] which are the
  //       log terms due to the prior over all possible sequences
  float smallnum = 1e-12;

  int num_sequences = sequences.size();
  std::vector<float> a_terms(num_sequences, 0.0f);

  for (int seq = 0; seq < num_sequences; seq++) {
    int day_s = sequences[seq][0];
    int day_e = sequences[seq][1];
    int day_i = sequences[seq][2];

    // Prior for starting in S, could be zero
    float a_term = std::log(1. - probab0 + smallnum) * day_s;

    if (day_e > 0) {
      a_term += std::log(probab0);

      a_term += std::log(1. - g_param) * (day_e-1);
    }

    if ((day_s == 0) & (day_e == 0))
      a_term += std::log(smallnum);

    if (day_i > 0) {
      if (day_e > 0)
        a_term += std::log(g_param);
      a_term += std::log(1. - h_param) * (day_i-1);
    }

    if ((day_s + day_e + day_i < num_time_steps) & (day_i > 0)) {
      a_term += std::log(h_param);
    }

    a_terms[seq] = a_term;
  }
  return a_terms;
}


array2<float> precompute_c_terms(
    const int& num_users,
    const float& alpha,
    const float& beta,
    const py::array_t<int>& observations_,
    const array3<int>& seq_hot) {
  // Precomputes the c-terms for the FN update.
  //
  // Args:
  //   num_users: number of users
  //   alpha: False positive rate
  //   beta: False negative rate
  //   observations_: array with observations, array in [num_observations, 3]
  //   seq_hot: 3d array with sequences, array in [num_time_steps, 4, num_seq]
  //
  // Returns:
  //   c_terms: 2d array with c_terms, in [num_users, num_sequences]
  assert(observations.size() > 0);
  assert(observations[0].size() == 3);

  int num_sequences = seq_hot[0][0].size();
  std::vector<std::vector<float>> c_terms(
    num_users, std::vector<float>(num_sequences, 0.0));

  auto observations = observations_.unchecked<2>();
  int num_observations = observations.shape(0);

  for (int obs = 0; obs < num_observations; obs++) {
    int user = observations(obs, 0);
    int timestep = observations(obs, 1);
    int outcome = observations(obs, 2);

    float penalty = 0.0f;
    int is_inf = 0;

    for (int seq = 0; seq < num_sequences; seq++) {
      is_inf = seq_hot[timestep][2][seq];

      if (is_inf == 1) {
        penalty = outcome ? 1-alpha : alpha;
      } else {
        penalty = outcome ? beta : 1-beta;
      }

      c_terms[user][seq] += std::log(penalty);
    }
  }
  return c_terms;
}


int calc_softmax(
  const float* logit,
  const int& num_elements,
  float* distr
) {
  float max_logit = logit[0];
  for (int i = 1; i < num_elements; i++) {
    if (logit[i] > max_logit)
      max_logit = logit[i];
  }

  float sum = 0.0;
  for (int i = 0; i < num_elements; i++) {
    sum += std::exp(logit[i] - max_logit);
  }

  for (int i = 0; i < num_elements; i++) {
    distr[i] = std::exp(logit[i] - max_logit) / sum;
  }

  return 0;
}


int fn_update(
    const int& user_start,
    const int& user_end,
    const int& num_time_steps,
    const float& probab0,
    const float& probab1,
    const float& rho_rdp,
    const float& a_rdp,
    const float& clip_lower,
    const float& clip_upper,
    const array3<int>& seq_array_hot,
    const array2<float>& log_c_z_u,
    const array1<float>& log_a_start,
    const array2<float>& p_infected,
    const int * past_contacts,
    array3<float>& post_exp,
    const int dedup_contacts = 0) {
  // Makes one update step for FN
  //
  // Args:
  //   user_start: start index of users
  //   user_end: end index of users
  //   num_time_steps: number of time steps
  //   probab0: probability of starting in state 0
  //   probab1: probability transit S->E for an infected contact
  //   rho_rdp: rho privacy parameter for RDP
  //   a_rdp: a privacy parameter for RDP
  //   clip_lower_: lower bound for clipping
  //   clip_upper_: upper bound for clipping
  //   seq_array_hot: 3d array with sequences,
  //       array in [num_time_steps, 4, num_seq]
  //  log_c_z_u: 2d array with c_terms, in [num_users, num_sequences]
  //  log_a_start: vector of a_terms, of length [num_sequences] which are the
  //       log terms due to the prior over all possible sequences
  //  p_infected: 2d array with p_infected, in [num_users, num_time_steps]
  //  past_contacts: past contacts, array in [num_users, num_contacts, 2]
  //  post_exp: 3d array with post_exp, in [num_users, num_time_steps, 4]
  //      this array will be modified in-place
  //
  // Returns:
  //   integer 1 if succesful finished
  assert(post_exp[0][0][0] == 0.f);

  int num_sequences = seq_array_hot[0][0].size();
  assert(log_a_start.size() == num_sequences);
  assert(seq_array_hot.size() == num_time_steps);

  int num_days_s[num_sequences] = {0};
  // Set the start_belief for each sequence
  for (int seq = 0; seq < num_sequences; seq++) {
      // Find out the numbers of days spent in s
      // For calculation of d_terms later
      for (int t = 0; t < num_time_steps; t++) {
        num_days_s[seq] += seq_array_hot[t][0][seq];
      }
  }

  for (int num_user = user_start; num_user < user_end; num_user++) {
      // Compute d terms
      // Take d terms

      array2<float> d_terms = precompute_d_terms(
        num_user,
        num_time_steps,
        p_infected,
        past_contacts,
        probab0,
        probab1,
        rho_rdp,
        a_rdp,
        clip_lower,
        clip_upper,
        dedup_contacts);

      // Calculate d_notermcumsum
      float d_noterm_cumsum[num_time_steps+1] = {0.};
      // TODO(rob): check element 0 here
      for (int t = 0; t < num_time_steps; t++) {
        d_noterm_cumsum[t+1] = d_noterm_cumsum[t] + d_terms[1][t+1];
      }

      float d_penalties[num_sequences + 1] = {0.};
      for (int seq = 0; seq < num_sequences; seq++) {
        if (num_days_s[seq] > 0) {
          d_penalties[seq] += d_terms[0][num_days_s[seq]];
        }
        if (num_days_s[seq] > 1) {
          d_penalties[seq] += d_noterm_cumsum[num_days_s[seq]-1];
        }
      }
      // Get log joint
      float log_joint[num_sequences] = {0.};
      for (int seq = 0; seq < num_sequences; seq++) {
        log_joint[seq] = (
          log_c_z_u[num_user][seq]  // log terms due to observations
          + d_penalties[seq]  // log terms due to contacts (potentially DP)
          + log_a_start[seq]);  // log terms due to prior over sequences
      }

      float distr[num_sequences] = {0.};
      calc_softmax(log_joint, num_sequences, distr);

      // Calc post exp
      // TODO(rob): fix this with matrix library for more speedup!
      for (int t = 0; t < num_time_steps; t++) {
        for (int state = 0; state < 4; state++) {
          post_exp[num_user][t][state] = 0.0;
          for (int seq = 0; seq < num_sequences; seq++) {
            post_exp[num_user][t][state] += (
              distr[seq] * seq_array_hot[t][state][seq]);
          }
        }
      }
  }
  return 0;
  }


py::array_t<float> fn_wrapped(
    const int& user_start,
    const int& user_end,
    const int& num_time_steps,
    const float& probab0,
    const float& probab1,
    const float& rho_rdp,
    const float& a_rdp,
    const float& clip_lower,
    const float& clip_upper,
    const array3<int>& seq_array_hot,
    const array2<float>& log_c_z_u,
    const array1<float>& log_a_start,
    const array2<float>& p_infected,
    const py::array_t<int>& contacts) {
  // Wraps the FN update for use from Python
  //
  // Use this function only for unit-testing.
  // This function does a copy which is slow

  int num_users = user_end - user_start;

  // Make post_exp a 3d array
  std::vector<std::vector<std::vector<float>>> post_exp(
    num_users, std::vector<std::vector<float>>(
      num_time_steps, std::vector<float>(4, 0.0)));
  std::vector<std::vector<std::vector<float>>>& post_exp_ref {post_exp};

  // Move past contacts to array
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  fn_update(
    user_start,
    user_end,
    num_time_steps,
    probab0,
    probab1,
    rho_rdp,
    a_rdp,
    clip_lower,
    clip_upper,
    seq_array_hot,
    log_c_z_u,
    log_a_start,
    p_infected,
    past_contacts,
    post_exp_ref);

  // Put post_exp back to 3d array
  delete[] past_contacts;
  return py::array(py::cast(post_exp));
  }


array2<int> iter_sequences(int num_time_steps) {
  // Iterates over all possible sequences in num_t time steps.
  //
  // Args:
  //   num_time_steps: number of time steps
  //
  // Returns:
  //   sequences: 2d array with all possible sequences, in [num_contacts, 3]
  array2<int> sequences;
  for (int day_s = 0; day_s < num_time_steps+1; day_s++) {
    if (day_s == num_time_steps) {
      sequences.push_back({day_s, 0, 0});
    } else {
      int e_start = (day_s > 0) ? 1 : 0;
      for (int day_e = e_start; day_e < num_time_steps-day_s+1; day_e++) {
        if (day_s+day_e == num_time_steps) {
          sequences.push_back({day_s, day_e, 0});
        } else {
          int i_start = (day_s > 0 || day_e > 0) ? 1 : 0;
          for (int day_i = i_start; day_i < num_time_steps-day_s-day_e+1;
                 day_i++) {
            sequences.push_back({day_s, day_e, day_i});
          }
        }
      }
    }
  }
  return sequences;
}


array3<int> make_seq_array_hot(
    const array2<int>& sequences, const int& num_time_steps) {
  // Makes a 3d array of sequences with one-hot values
  //
  // sequences is a list of lists
  // sequences = [[day_s, day_e, day_i], [day_s, day_e, day_i], ...]
  // day_s is the number of time steps spent in state 0
  // day_e is the number of time steps spent in state 1
  // day_i is the number of time steps spent in state 2
  //
  // Args:
  //   sequences: 2d array with all possible sequences, in [num_contacts, 3]
  //   num_time_steps: number of time steps
  //
  // Returns:
  //   A 3d-array of size [num_time_steps, 4, num_sequences]
  int num_sequences = sequences.size();

  array1<int> row(num_sequences, 0);
  std::vector<std::vector<int>> column(4, row);
  std::vector<std::vector<std::vector<int>>> seq_hot(num_time_steps, column);

  for (int seq = 0; seq < num_sequences; seq++) {
    int day_s = sequences[seq][0];
    int day_e = sequences[seq][1];
    int day_i = sequences[seq][2];

    for (int t = 0; t < day_s; t++) {
      seq_hot[t][0][seq] = 1;
    }
    for (int t = day_s; t < day_s+day_e; t++) {
      seq_hot[t][1][seq] = 1;
    }
    for (int t = day_s+day_e; t < day_s+day_e+day_i; t++) {
      seq_hot[t][2][seq] = 1;
    }
    for (int t = day_s+day_e+day_i; t < num_time_steps; t++) {
      seq_hot[t][3][seq] = 1;
    }
  }

  return seq_hot;
}


int save_age_contacts(
  const int& num_users,
  const int * past_contacts,
  const py::array_t<int>& users_age_pyarray,
  std::vector<int>& age_contacts) {
  // Saves the quantiles of contacts' ages

  auto users_age = users_age_pyarray.unchecked<1>();
  int idx, user_send;
  float q50, q80;  // 50th and 80th percentile

  for (int user = 0; user < num_users; user++) {
    std::vector<int> ages;
    for (int t = 0; t < CTC_SIZE; t++) {
      idx = user*CTC_SIZE*2 + t*2;
      if (past_contacts[idx] < 0) {
        break;}

      user_send = past_contacts[idx+1];
      ages.push_back(users_age(user_send));
    }

    if (ages.size() > 0) {
      std::sort(ages.begin(), ages.end());

      int idx50 = static_cast<int>(std::floor(0.5 * ages.size()));
      int idx80 = static_cast<int>(std::floor(0.79 * ages.size()));

      q50 = ages[idx50];
      q80 = ages[idx80];
    } else {
      q50 = -1.f;
      q80 = -1.f;
    }

    age_contacts[user] = q50;
    age_contacts[user+num_users] = q80;
  }

  return 0;
}


std::pair<array3<float>, std::vector<int>> fn_full(
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
    const int dedup_contacts) {
  // Repeats the FN update for num_rounds iterations
  //
  // Args:
  //   num_workers: number of workers
  //   num_rounds: number of rounds
  //   num_users: number of users
  //   num_time_steps: number of time steps
  //   probab0: probability of starting in state 0
  //   probab1: probability transit S->E for an infected contact
  //   g_param: probability of transition E->I
  //   h_param: probability of transition I->R
  //   alpha: False positive rate
  //   beta: False negative rate
  //   rho_rdp: rho privacy parameter for RDP
  //   a_rdp: a privacy parameter for RDP
  //   clip_lower: lower bound for clipping
  //   clip_upper: upper bound for clipping
  //   observations: array with observations, array in [num_observations, 3]
  //   contacts: array with contacts, array in [num_contacts, 4]
  //
  // Returns:
  //   post_exp: 3d array with the posterior estimates,
  //       array in [num_users, num_time_steps, 4]

  // We will not modify any python objects, so we can release the GIL
  std::chrono::time_point<std::chrono::system_clock> m_StartTime =
    std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds;

  py::gil_scoped_release release;

  // Initialize p_inf
  std::vector<std::vector<float>> p_inf(
    num_users, std::vector<float>(num_time_steps, 0.0));
  std::vector<std::vector<float>>& p_inf_ref {p_inf};

  // Initialize post_exp
  std::vector<std::vector<std::vector<float>>> post_exp(
    num_users, std::vector<std::vector<float>>(
      num_time_steps, std::vector<float>(4, 0.0)));
  std::vector<std::vector<std::vector<float>>>& post_exp_ref {post_exp};

  // Get contacts
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  std::vector<int> age_contacts(2*num_users, 0);
  std::vector<int>& age_contacts_ref {age_contacts};
  save_age_contacts(
    num_users, past_contacts, users_age, age_contacts_ref);

  std::thread save_age_thread([&]() {
    save_age_contacts(num_users, past_contacts, users_age, age_contacts_ref);
    });

  // Initialize seq_array_hot
  array2<int> sequences = iter_sequences(num_time_steps);
  array3<int> seq_array_hot = make_seq_array_hot(sequences, num_time_steps);

  // Get a terms
  std::vector<float> log_a_array =  precompute_a_terms(
    num_time_steps, probab0, g_param, h_param, sequences);

  // Get c terms
  array2<float> log_c_terms = precompute_c_terms(
    num_users, alpha, beta, observations, seq_array_hot);

  // Split users among cores
  std::vector<int> data_array = divide_work(num_users, num_workers);
  std::vector<int> user_id(num_workers+1, 0);
  for (int i = 1; i < num_workers+1; i++) {
    user_id[i] = user_id[i-1] + data_array[i-1];
  }

  elapsed_seconds = std::chrono::system_clock::now() - m_StartTime;
  std::cout << "Preamble time: " << elapsed_seconds.count() << "s\n";

  float rho_rdp_set = -1.;
  float a_rdp_set = -1.;
  float clip_lower_set = -1.;
  float clip_upper_set = 10.;

  float message = 0.0;
  for (int num_round = 0; num_round < num_rounds; num_round++) {
    if (num_round == num_rounds - 1) {
      // Only run DP on last round
      rho_rdp_set = rho_rdp;
      a_rdp_set = a_rdp;
      clip_lower_set = clip_lower;
      clip_upper_set = clip_upper;
    }

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
        fn_update(
          user_id[num_worker],
          user_id[num_worker+1],
          num_time_steps,
          probab0,
          probab1,
          rho_rdp_set,  // only pass value on last round
          a_rdp_set,  // only pass value on last round
          clip_lower_set,  // only pass value on last round
          clip_upper_set,  // only pass value on last round
          seq_array_hot,
          log_c_terms,
          log_a_array,
          p_inf_ref,
          past_contacts,
          post_exp_ref,
          dedup_contacts);
        });
      }

    for (int num_worker = 0; num_worker < num_workers; ++num_worker) {
      tlist[num_worker].join();
    }
  }
  save_age_thread.join();

  delete[] past_contacts;
  return std::pair<array3<float>, std::vector<int>>(post_exp, age_contacts);
}

array3<int> fn_data_dump(
    const int& num_workers,
    const int& num_users,
    const int& num_time_steps,
    const py::array_t<float>& q_marginal_pyarray,
    const py::array_t<int>& contacts,
    const py::array_t<int>& users_age_pyarray) {

  // Get contacts
  int * past_contacts = new int[num_users*CTC_SIZE*2];
  contact_flip(num_users, contacts, past_contacts);

  auto users_age = users_age_pyarray.unchecked<1>();
  auto q_marginal = q_marginal_pyarray.unchecked<2>();

  // For every contact, get the:
  // * timestep of the contact
  // * id of the sender
  // * Age of the sender
  // * COVIDSCORE of the sender

  int * datadump = new int[num_users*CTC_SIZE*NUM_DUMP_FEATURES];
  std::fill_n(datadump, num_users*CTC_SIZE*NUM_DUMP_FEATURES, -1);

  int sender, idx, pinf, idx_datadump, timestep;

  for (int num_user = 0; num_user < num_users; num_user++) {
    for (int num_contact = 0; num_contact < CTC_SIZE; num_contact++) {
      idx = num_user*CTC_SIZE*2 + num_contact*2;
      timestep = past_contacts[idx];
      if (timestep < 0)
          break;

      sender = past_contacts[idx+1];
      pinf = static_cast<int>(q_marginal(sender, timestep) * 1024);

      idx_datadump = (
        num_user*CTC_SIZE*NUM_DUMP_FEATURES + num_contact*NUM_DUMP_FEATURES);
      datadump[idx_datadump] = timestep;
      datadump[idx_datadump+1] = sender;
      datadump[idx_datadump+2] = users_age(sender);
      datadump[idx_datadump+3] = pinf;
    }
  }

  // Write datadump array to data_vector in std::vector format
  std::vector<std::vector<std::vector<int>>> data_vector(
    num_users, std::vector<std::vector<int>>(
      CTC_SIZE, std::vector<int>(NUM_DUMP_FEATURES, -1)));

  for (int num_user = 0; num_user < num_users; num_user++) {
    for (int num_contact = 0; num_contact < CTC_SIZE; num_contact++) {
      idx = (
        num_user*CTC_SIZE*NUM_DUMP_FEATURES+num_contact*NUM_DUMP_FEATURES);
      for (int i = 0; i < NUM_DUMP_FEATURES; i++) {
        data_vector[num_user][num_contact][i] = datadump[idx+i];
      }
    }
  }

  delete[] past_contacts;
  delete[] datadump;

  return data_vector;
}

#endif  // SRC_FN_HPP_
