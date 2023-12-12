"""Speed test the numba past contacts versus cpp implementation."""
import numba
import numpy as np
import timeit
from dpfn import util
import dpfn_util


def speed_past_contacts():
  """Tests speed of past contacts listing."""
  fname = "data/dummy_14_01.npy"
  num_users_orig = 100000
  contacts_all = np.load(fname)

  fname = "data/dummy_14_01_obs.npy"
  observations_all = np.load(fname)

  assert np.all(contacts_all[:, 2] <= 13)
  assert np.all(observations_all[:, 0] < num_users_orig)

  num_doubles = 2
  for i in range(num_doubles):
    contacts_all = np.concatenate([contacts_all, contacts_all], axis=0)
    num_half = contacts_all.shape[0] // 2
    contacts_all[num_half:, :2] += num_users_orig*2**i

    observations_all = np.concatenate(
      [observations_all, observations_all], axis=0)
    observations_all[num_half:, 0] += num_users_orig*2**i
  num_users = num_users_orig * 2**num_doubles

  num_contacts = np.sum(contacts_all[:, 0] > 0)
  print(f"Num users: {num_users}")
  print(f"Num contacts: {num_contacts}")

  num_time_steps = 14
  num_max_msg = int(num_time_steps*100)

  def option1():
    past_contacts, _ = util.get_past_contacts_static(
      (0, num_users), contacts_all, num_msg=num_max_msg)

    assert past_contacts[0][0][0] >= 0

  def option2():
    data = -1*np.ones((num_users, num_max_msg, 4), dtype=np.int32)
    dpfn_util.contact_flip(contacts=contacts_all, data=data)

    assert data[0][0][0] >= 0

  # JIT compile first
  option2()
  print("Pybind version")
  print(timeit.timeit(option2, number=2))

  option1()
  print("Numba version")
  print(timeit.timeit(option1, number=2))


if __name__ == '__main__':
  numba.set_num_threads(3)
  speed_past_contacts()
