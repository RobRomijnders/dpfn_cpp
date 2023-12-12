"""Speed test of calculating d terms against the original dpfn library."""
import numpy as np
import timeit
from dpfn import util
import dpfn_util


def speed_cpp_implementation_small():
  """Tests speed on a small graph of 6 users and 8 time steps."""
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 2, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8
  p0, p1 = 0.001, 0.1

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  def option1():
    _, _ = util.precompute_d_penalty_terms_fn2(
      q_marginal_infected,
      p0=p0,
      p1=p1,
      past_contacts=past_contacts[user],
      num_time_steps=num_time_steps)

  def option2():
    dpfn_util.d_terms_wrapped(
      num_user=user,
      num_time_steps=num_time_steps,
      q_marginal=q_marginal_infected,
      contacts=contacts_all,
      p0=p0,
      p1=p1,
      rho_rdp=-1.,
      a_rdp=-1.,
      clip_lower=-1.,
      clip_upper=10.,
    )

  # JIT compile first
  option1()
  option2()

  print(timeit.timeit(option1, number=1000))
  print(timeit.timeit(option2, number=1000))


def speed_cpp_implementation_large():
  """Tests the speed on a larger contact graph."""
  fname = (
    "/home/rob/Documents/phd_code/dpfn/results/trace_prequential/"
    "intermediate_graph_abm_02__model_ABM01/dummy_10k.npy")
  num_users = 10000

  fname = (
    "/home/rob/Documents/phd_code/dpfn/results/trace_prequential/"
    "large_graph_abm_01__model_ABM01/dummy_14_01.npy")
  num_users = 100000

  contacts_all = np.load(fname)

  num_time_steps = 14

  p0, p1 = 0.001, 0.1

  # Second test case
  user = 1
  q_infected = np.random.rand(num_users, num_time_steps).astype(np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  def option1():
    _, _ = util.precompute_d_penalty_terms_fn2(
      q_infected,
      p0=p0,
      p1=p1,
      past_contacts=past_contacts[user],
      num_time_steps=num_time_steps)

  def option2():
    dpfn_util.d_terms_wrapped(
      num_user=user,
      num_time_steps=num_time_steps,
      q_marginal=q_infected,
      contacts=contacts_all,
      p0=p0,
      p1=p1,
      rho_rdp=-1.,
      a_rdp=-1.,
      clip_lower=-1.,
      clip_upper=10.,
    )

  # JIT compile first
  option1()
  option2()

  print(timeit.timeit(option1, number=1000))
  print(timeit.timeit(option2, number=1000))


if __name__ == '__main__':
  speed_cpp_implementation_large()
