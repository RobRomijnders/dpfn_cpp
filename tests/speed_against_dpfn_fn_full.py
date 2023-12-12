"""Speed test the full fn_wrapped against the original dpfn library."""
from contextlib import contextmanager
import numba
import numpy as np
from dpfn import logger, inference, util
import dpfn_util
import os
import psutil
import threading
import time
from time import perf_counter


@contextmanager
def catchtime(message="Time"):
  start = perf_counter()
  yield
  logger.info(f"{message}: {perf_counter() - start:.2f} s")


def log_memory():
  """Logs memory usage of this process."""
  fname = "log/memory_usage.txt"
  with open(fname, "w") as f:
    f.write("Memory usage:")

  while True:
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    loadavg, _, _ = os.getloadavg()
    with open(fname, "a") as f:
      f.write(f"Mem: {mem:10.1f}, loadavg: {loadavg:5.1f}\n")
    time.sleep(1)


def speed_fn_full(num_workers: int = 1):
  """Tests speed of the full fn function against the original dpfn library."""
  daemon_logger = threading.Thread(target=log_memory, args=(), daemon=True)
  daemon_logger.start()

  fname_c = "data/dummy_14_01.npy"
  fname_o = "data/dummy_14_01_obs.npy"
  num_users_orig = 100000

  # fname_c = "data/contacts_10k.npy"
  # fname_o = "data/observations_10k.npy"
  # num_users_orig = 10000

  contacts_all = np.load(fname_c)
  observations_all = np.load(fname_o)

  assert np.all(contacts_all[:, 2] <= 13)
  assert np.all(observations_all[:, 0] < num_users_orig)

  logger.info(f"Dtype contacts: {contacts_all.dtype}")
  logger.info(f"Dtype observations: {observations_all.dtype}")

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
  logger.info(f"Num contacts: {num_contacts}")

  num_time_steps = 14
  probab0, probab1 = 0.001, 0.1
  g_param, h_param = 0.99, 0.10
  alpha, beta = 0.001, 0.01

  def option_numba():
    post_exp = inference.fact_neigh(
      num_users=num_users,
      num_updates=5,
      num_time_steps=num_time_steps,
      observations_all=observations_all,
      contacts_all=contacts_all,
      probab_0=probab0,
      probab_1=probab1,
      g_param=g_param,
      h_param=h_param,
      alpha=alpha,
      beta=beta,
      quantization=128)

    assert np.all(post_exp >= 0.)
    assert np.all(post_exp <= 1.001)
    return post_exp

  def option_pybind():
    post_exp_out = dpfn_util.fn_full_func(
      num_workers=num_workers,
      num_rounds=5,
      num_users=num_users,
      num_time_steps=num_time_steps,
      probab0=probab0,
      probab1=probab1,
      g_param=g_param,
      h_param=h_param,
      alpha=alpha,
      beta=beta,
      rho_rdp=-1.,
      a_rdp=-1.,
      clip_lower=-1.,
      clip_upper=10.,
      quantization=128,
      observations=observations_all,
      contacts=contacts_all)

    assert np.all(post_exp_out >= 0.)
    assert np.all(post_exp_out <= 1.001)
    return post_exp_out

  logger.info("Pybind version")
  with catchtime("Pybind"):
    post_exp_pybind = option_pybind()

  logger.info("Numba version")
  with catchtime("Numba JIT"):
    option_numba()

  with catchtime("Numba"):
    post_exp_numba = option_numba()

  np.save("log/post_exp_pybind.npy", post_exp_pybind)
  np.save("log/post_exp_numba.npy", post_exp_numba)

  np.testing.assert_array_almost_equal(
    post_exp_pybind, post_exp_numba, decimal=3)


if __name__ == '__main__':
  num_threads = max((util.get_cpu_count()-1, 1))
  num_threads = min((num_threads, 8))  # Many threads are not faster
  numba.set_num_threads(num_threads)
  logger.info(f"SLURM env N_TASKS: {os.getenv('SLURM_NTASKS')}")
  logger.info(f"Start with {num_threads} threads")

  speed_fn_full(num_workers=num_threads)
