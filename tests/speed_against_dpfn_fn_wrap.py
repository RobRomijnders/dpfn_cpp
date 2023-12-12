"""Speed test the full fn_wrapped against the original dpfn library."""
import numba
import numpy as np
import timeit
from dpfn import constants, inference, util
import dpfn_util
import os
import psutil
import threading
import time


def log_memory():
  """Logs memory usage of this process."""
  fname = "log/memory_usage.txt"
  with open(fname, "w") as f:
    f.write("Memory usage:")

  while True:
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    with open(fname, "a") as f:
      f.write(f"{mem:10.1f}\n")
    time.sleep(1)


def speed_cpp_implementation_large():
  """Tests speed of the full fn_wrapped against the original dpfn library."""
  fname_c = "data/dummy_14_01.npy"
  fname_o = "data/dummy_14_01_obs.npy"
  num_users_orig = 100000

  contacts_all = np.load(fname_c)
  observations_all = np.load(fname_o)

  print(f"Num contacts before: {len(contacts_all)}")

  assert np.all(contacts_all[:, 2] <= 13)
  assert np.all(observations_all[:, 0] < num_users_orig)

  num_doubles = 0
  for i in range(num_doubles):
    contacts_all = np.concatenate([contacts_all, contacts_all], axis=0)
    num_half = contacts_all.shape[0] // 2
    contacts_all[num_half:, :2] += num_users_orig*2**i

    observations_all = np.concatenate(
      [observations_all, observations_all], axis=0)
    observations_all[num_half:, 0] += num_users_orig*2**i
  num_users = num_users_orig * 2**num_doubles

  num_contacts = np.sum(contacts_all[:, 0] > 0)
  print(f"Num contacts: {num_contacts}")

  t_start_preamble = time.time()

  num_time_steps = 14
  probab0, probab1 = 0.001, 0.1
  g_param, h_param = 0.99, 0.10
  alpha, beta = 0.001, 0.01

  # Second test case
  user_interval = (0, num_users)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  print(
    f"Theoretical usage of pc: {past_contacts.nbytes/1024**3:.1f} GB")
  size = np.prod(past_contacts.shape) * 4 / 1024**3
  print(f"Calculated usage of pc: {size:.1f} GB")

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int32)

  prior = np.array([1-probab0, probab0, 0., 0.], dtype=np.float32)

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  # Precompute log(C) terms, relating to observations
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = np.random.rand(num_users, num_time_steps)
  q_marginal_infected = q_marginal_infected.astype(np.float32)

  t_preamble1 = time.time() - t_start_preamble
  t_start_preamble = time.time()

  num_max_msg = int(constants.CTC*max((num_time_steps, 14)))
  past_contacts = dpfn_util.contact_flip(
    num_users=num_users, max_msg=num_max_msg, contacts=contacts_all)

  t_preamble2 = time.time() - t_start_preamble
  print(
    f"Time spent on preamble1/preamble2 {t_preamble1:.1f}/{t_preamble2:.1f}")

  daemon_logger = threading.Thread(target=log_memory, args=(), daemon=True)
  daemon_logger.start()

  def option1():
    post_exp, _, _ = inference.fn_step_wrapped(
      user_interval,
      seq_array_hot,
      log_c_z_u,  # already depends in mpi_rank
      log_A_start,
      q_marginal_infected,
      num_time_steps,
      probab0,
      probab1,
      clip_lower=-1.,
      clip_upper=10000.,
      past_contacts_array=past_contacts,
      dp_method=-1,
      epsilon_dp=-1.,
      delta_dp=-1.,
      quantization=128)

    assert np.all(post_exp >= 0.)
    assert np.all(post_exp <= 1.001)

  def option2():
    _ = dpfn_util.fn_wrapped_func(
      user_start=0,
      user_end=num_users,
      num_time_steps=num_time_steps,
      probab0=probab0,
      probab1=probab1,
      rho_rdp=-1.,
      a_rdp=-1.,
      clip_lower=-1.,
      clip_upper=10.,
      seq_array_hot=seq_array_hot,
      log_c_z_u=log_c_z_u,
      log_a_start=log_A_start,
      p_infected_matrix=q_marginal_infected,
      past_contacts=past_contacts)

  # JIT compile first
  option2()
  print("Pybind version")
  print(timeit.timeit(option2, number=2))

  option1()
  print("Numba version")
  print(timeit.timeit(option1, number=2))


if __name__ == '__main__':
  numba.set_num_threads(3)
  speed_cpp_implementation_large()
