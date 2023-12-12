"""Test utility functions related to Belief Propagation."""
import numpy as np
import dpfn_util


def test_make_obs_messages():
  num_users = 13
  num_time_steps = 7
  alpha = 0.0001
  beta = 0.001

  observations = np.array([
    (0, 2, 1),
    [0, 3, 0],
    [1, 2, 1],
    [1, 3, 1],
    [2, 3, 1],
  ], dtype=np.int32)

  obs_messages = dpfn_util.make_obs_messages(
    num_users=num_users,
    num_time_steps=num_time_steps,
    alpha=alpha,
    beta=beta,
    observations=observations
  ).reshape((num_users, num_time_steps, 4))

  np.testing.assert_almost_equal(obs_messages[0, 3, 0], 1-beta, decimal=3)
  np.testing.assert_almost_equal(obs_messages[0, 3, 2], alpha, decimal=3)
  np.testing.assert_almost_equal(obs_messages[0, 0, 0], 1, decimal=3)

  np.testing.assert_almost_equal(obs_messages[1, 2, 0], beta, decimal=3)
  np.testing.assert_almost_equal(obs_messages[1, 2, 2], 1-alpha, decimal=3)


def test_forward_backward_dry():
  num_users = 13
  num_time_steps = 7

  observations = np.array([
    [1, 2, 1],
    [1, 3, 1],
    [2, 3, 1],
  ], dtype=np.int32)

  contacts = np.array([
    [1, 2, 1, 1]
  ], dtype=np.int32)

  post_exp = dpfn_util.forward_backward_user_wrapped(
    num_user=0,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=0.01,
    probab1=0.01,
    g_param=0.99,
    h_param=0.2,
    alpha=0.0001,
    beta=0.001,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    p_infected=np.zeros((num_users, num_time_steps), dtype=np.float32),
    observations=observations,
    contacts=contacts)

  assert post_exp.shape == (num_users, num_time_steps, 4)
  np.testing.assert_array_less(0.9, post_exp[0][-1][0])
  assert post_exp[0][-1][3] < 0.1


def test_forward_backward_with_test():
  num_users = 13
  num_time_steps = 7

  observations = np.array([
    [0, 2, 1],
    [1, 3, 1],
    [2, 3, 1],
  ], dtype=np.int32)

  contacts = np.array([
    [1, 2, 1, 1]
  ], dtype=np.int32)

  post_exp = dpfn_util.forward_backward_user_wrapped(
    num_user=0,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=0.01,
    probab1=0.01,
    g_param=0.99,
    h_param=0.2,
    alpha=0.0001,
    beta=0.001,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    p_infected=np.zeros((num_users, num_time_steps), dtype=np.float32),
    observations=observations,
    contacts=contacts
  )

  assert post_exp.shape == (num_users, num_time_steps, 4)
  assert post_exp[0][-1][0] < 0.1
  assert post_exp[0][-1][3] > 0.2


def test_forward_backward_with_infected_contact():
  num_users = 13
  num_time_steps = 7

  observations = np.array([
    [2, 3, 1],
  ], dtype=np.int32)

  contacts = np.array([
    [1, 0, 1, 1],
    [1, 2, 1, 1],
    [1, 3, 1, 1]
  ], dtype=np.int32)

  post_exp_before = dpfn_util.forward_backward_user_wrapped(
    num_user=0,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=0.01,
    probab1=0.3,
    g_param=0.99,
    h_param=0.2,
    alpha=0.0001,
    beta=0.001,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    p_infected=np.zeros((num_users, num_time_steps), dtype=np.float32),
    observations=observations,
    contacts=contacts
  )

  post_exp_after = dpfn_util.forward_backward_user_wrapped(
    num_user=0,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=0.01,
    probab1=0.3,
    g_param=0.99,
    h_param=0.2,
    alpha=0.0001,
    beta=0.001,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    p_infected=np.ones((num_users, num_time_steps), dtype=np.float32),
    observations=observations,
    contacts=contacts
  )

  np.testing.assert_array_less(
    post_exp_before[0, 3:, 2], post_exp_after[0, 3:, 2])
  np.testing.assert_array_less(0.15, post_exp_after[0][-1][2])
  np.testing.assert_array_less(0.2, post_exp_after[0][2][1])


def test_bp_full():
  num_users = 13
  num_time_steps = 7

  observations = np.array([
    [1, 3, 1],
    [2, 3, 1],
  ], dtype=np.int32)

  contacts = np.array([
    [1, 0, 1, 1],
    [1, 2, 1, 1],
    [2, 4, 1, 1]
  ], dtype=np.int32)

  post_exp = dpfn_util.bp_full_func(
    num_workers=2,
    num_rounds=5,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=0.01,
    probab1=0.4,
    g_param=0.99,
    h_param=0.2,
    alpha=0.0001,
    beta=0.001,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=0.00,
    clip_upper=0.99,
    quantization=128,
    observations=observations,
    contacts=contacts
  )

  np.testing.assert_array_less(
    post_exp[5, 3:, 2], post_exp[4, 3:, 2])  # User 4 has infected contact
  np.testing.assert_array_less(0.4, post_exp[2][3][2])
  np.testing.assert_array_less(0.1, post_exp[0][3][2])
