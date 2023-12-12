"""Test against the original dpfn library."""
import numpy as np
from dpfn import constants, util
import dpfn_util

# def test_numpy_change():
#   data = np.zeros((5, 5), dtype=np.float32)
#   dpfn_util.numpy_change(data)


def test_divide_work():

  data = dpfn_util.divide_work(num_jobs=11, num_workers=5)
  assert np.issubdtype(data.dtype, np.integer)
  np.testing.assert_array_almost_equal(
    data, np.array([3, 2, 2, 2, 2], dtype=np.int32))

  data = dpfn_util.divide_work(num_jobs=12, num_workers=5)
  np.testing.assert_array_almost_equal(
    data, np.array([3, 3, 2, 2, 2], dtype=np.int32))

  data = dpfn_util.divide_work(num_jobs=3, num_workers=5)
  np.testing.assert_array_almost_equal(
    data, np.array([1, 1, 1, 0, 0], dtype=np.int32))


def test_contact_flip():
  num_users = 10

  contacts = np.array([
    [1, 2, 3, 1],
    [3, 2, 6, 1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1]], dtype=np.int32)

  past_contacts = dpfn_util.contact_flip(
    num_users=num_users, contacts=contacts)
  past_contacts = np.reshape(past_contacts, [num_users, constants.CTC, 2])
  np.testing.assert_array_equal(
    past_contacts.shape, [num_users, constants.CTC, 2])


def test_iter_sequences():
  num_time_steps = 7

  sequences_dpfn = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))

  sequences_cpp = dpfn_util.iter_sequences(num_time_steps=num_time_steps)

  assert np.issubdtype(sequences_cpp.dtype, np.integer)

  np.testing.assert_array_almost_equal(
    sequences_dpfn, sequences_cpp)


def test_seq_array_hot():
  # Try a few values of num_time_steps
  for num_time_steps in [7, 13]:
    seq_array = np.stack(list(
      util.iter_sequences(time_total=num_time_steps, start_se=False)))
    seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
      seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int64)

    sequences_cpp = dpfn_util.iter_sequences(num_time_steps=num_time_steps)
    seq_hot_cpp = dpfn_util.make_seq_array_hot(
      sequences_cpp, num_time_steps=num_time_steps)

    assert np.issubdtype(seq_hot_cpp.dtype, np.integer)
    np.testing.assert_array_equal(
      seq_array_hot.shape, seq_hot_cpp.shape)
    np.testing.assert_array_equal(
      seq_array_hot, seq_hot_cpp)


def test_a_terms():
  num_time_steps = 7
  probab0 = 0.01
  g_param = 0.2
  h_param = 0.16

  sequences = dpfn_util.iter_sequences(num_time_steps=num_time_steps)

  # Calculate old way
  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))

  prior = [1-probab0, probab0, 0., 0.]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab0, 1-g_param, 1-h_param],
    seq_array, num_time_steps).astype(np.float32)

  # Calculate new way
  a_terms = dpfn_util.precompute_a_terms(
    num_time_steps=num_time_steps,
    probab0=probab0,
    g_param=g_param,
    h_param=h_param,
    sequences=sequences,
  )

  np.testing.assert_array_equal(log_A_start.shape, a_terms.shape)
  np.testing.assert_array_almost_equal(log_A_start, a_terms, decimal=5)


def test_c_terms():
  num_users = 13
  num_time_steps = 7
  alpha = 0.0001
  beta = 0.001

  observations_all = np.array([
    (0, 2, 1),
    [0, 3, 1],
    [1, 2, 1],
    [1, 3, 1],
    [2, 3, 1],
  ], dtype=np.int32)

  sequences = dpfn_util.iter_sequences(num_time_steps=num_time_steps)
  seq_hot = dpfn_util.make_seq_array_hot(
    sequences, num_time_steps=num_time_steps)

  # Precompute log(C) terms, relating to observations
  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  log_c_z_u = util.calc_c_z_u(
    (0, num_users),
    obs_array,
    observations_all)

  c_terms = dpfn_util.precompute_c_terms(
    num_users=num_users,
    alpha=alpha,
    beta=beta,
    observations=observations_all,
    seq_hot=seq_hot)

  np.testing.assert_array_equal(log_c_z_u.shape, c_terms.shape)

  np.testing.assert_array_almost_equal(log_c_z_u, c_terms, decimal=5)
