#ifndef SRC_UTIL_HPP_
#define SRC_UTIL_HPP_

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


void set_negative(int * past_contacts, int num_users) {
  for (int i = 0; i < num_users*CTC_SIZE*2; i++) {
    past_contacts[i] = -1;
  }
}


int contact_flip(
    const int& num_users,
    const py::array_t<int>& contacts,
    int * past_contacts) {
  // Flips the contacts from a long list to past_contacts per user
  //
  // Args:
  //   num_users: number of users
  //   contacts: array with contacts, array in [num_contacts, 3]
  //
  // Returns:
  //   int to indicate success
  set_negative(past_contacts, num_users);

  auto contacts_array = contacts.unchecked<2>();
  int n_contacts = contacts_array.shape(0);

  float num_contacts_per_user[num_users] = {0};
  int num_contact_overshoot = 0;
  int num_contact_user;
  int idx;
  int receiver;

  for (int n_contact = 0; n_contact < n_contacts; n_contact++) {
    receiver = contacts_array(n_contact, 1);

    if (receiver < 0)
      break;

    if (receiver > num_users) {
      std::cout << "Receiver index: " << receiver << std::endl;
      std::cout << "Receiver index: " << receiver << std::endl;
      std::cout << "num_users: " << num_users << std::endl;
      throw std::invalid_argument("Receiver index out of bounds");
    }

    num_contact_user = num_contacts_per_user[receiver];
    idx = receiver*CTC_SIZE*2 + num_contact_user*2;

    if (num_contact_user >= CTC_SIZE) {
      num_contact_overshoot += 1;
      continue;
    }

    // Timestep
    past_contacts[idx] = contacts_array(n_contact, 2);
    // Sender
    past_contacts[idx+1] = contacts_array(n_contact, 0);

    num_contacts_per_user[receiver] += 1;
  }

  if (num_contact_overshoot > 0) {
    std::cout << num_contact_overshoot << "/" << n_contacts <<
      " contacts were not stored due to CTC_SIZE" << std::endl;
  }

  return 1;
}

#endif  // SRC_UTIL_HPP_
