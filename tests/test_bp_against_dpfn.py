"""Test against the original dpfn library."""
import numpy as np
from dpfn import belief_propagation, util
import dpfn_util  # C++ implementation


def prepare_example():
  """Prepares a general example for testing."""
  contacts_all = np.array([
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (3, 2, 1, 1),
    (2, 3, 1, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ], dtype=np.int32)
  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  user_interval = (0, num_users)

  p0, p1 = 0.01, 0.3
  g_param, h_param = 0.2, 0.2
  alpha, beta = 0.001, 0.01

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.float32)

  prior = [1-p0, p0, 0., 0.]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-p0, 1-g_param, 1-h_param],
    seq_array, num_time_steps).astype(np.float32)

  # Precompute log(C) terms, relating to observations
  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  log_c_z_u = util.calc_c_z_u(
    user_interval,
    obs_array,
    observations_all)

  q_marginal_infected = 0.01 * np.random.rand(num_users, num_time_steps)
  q_marginal_infected = q_marginal_infected.astype(np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=num_time_steps*5)

  return (
    num_users, num_time_steps, p0, p1, seq_array_hot, log_c_z_u, log_A_start,
    past_contacts, q_marginal_infected, observations_all,
    contacts_all, g_param, h_param, alpha, beta)


def test_forward_backward_with_test():
  (num_users, num_time_steps, p0, p1, _, _, _, _, _, observations, contacts,
    g_param, h_param, alpha, beta) = prepare_example()

  post_exp = dpfn_util.forward_backward_user_wrapped(
    num_user=0,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=p0,
    probab1=p1,
    g_param=g_param,
    h_param=h_param,
    alpha=alpha,
    beta=beta,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    p_infected=np.zeros((num_users, num_time_steps), dtype=np.float32),
    observations=observations,
    contacts=contacts
  )

  # Original BP
  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-g_param, g_param, 0],
    [0, 0, 1-h_param, h_param],
    [0, 0, 0, 1]
  ], dtype=np.float32)

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta], dtype=np.float32),
    1: np.array([beta, beta, 1-alpha, beta], dtype=np.float32),
  }

  # Collect observations, allows for multiple observations per user per day
  obs_messages = np.ones((num_users, num_time_steps, 4), dtype=np.float32)
  for obs in observations:
    if obs[1] < num_time_steps:
      obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts, (0, num_users)))

  bp_belief, _, _ = belief_propagation.forward_backward_user(
    A_matrix=A_matrix,
    p0=p0,
    p1=p1,
    user=0,
    backward_messages=map_backward_message[0],
    forward_messages=map_forward_message[0],
    num_time_steps=num_time_steps,
    obs_messages=obs_messages[0],
    clip_lower=-1.,
    clip_upper=10.,
    epsilon_dp=100000.,  # With positive epsilon, bwd messages are frozen
    a_rdp=1.1)

  np.testing.assert_array_almost_equal(post_exp[0].shape, bp_belief.shape)
  np.testing.assert_array_almost_equal(bp_belief, post_exp[0], decimal=2)


def test_bp_full():
  (num_users, num_time_steps, p0, p1, _, _, _,
    _, _, observations_all, contacts_all,
    g_param, h_param, alpha, beta) = prepare_example()

  quantization = 128
  clip_upper = 0.9

  post_exp_bp = dpfn_util.bp_full_func(
    num_workers=1,
    num_rounds=5,
    num_users=num_users,
    num_time_steps=num_time_steps,
    probab0=p0,
    probab1=p1,
    g_param=g_param,
    h_param=h_param,
    alpha=alpha,
    beta=beta,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=clip_upper,
    quantization=quantization,
    observations=observations_all,
    contacts=contacts_all)

  # Original BP
  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-g_param, g_param, 0],
    [0, 0, 1-h_param, h_param],
    [0, 0, 0, 1]
  ], dtype=np.float32)

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta], dtype=np.float32),
    1: np.array([beta, beta, 1-alpha, beta], dtype=np.float32),
  }

  # Collect observations, allows for multiple observations per user per day
  obs_messages = np.ones((num_users, num_time_steps, 4), dtype=np.float32)
  for obs in observations_all:
    if obs[1] < num_time_steps:
      obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users)))

  for _ in range(5):
    (bp_beliefs, map_backward_message, map_forward_message, _) = (
      belief_propagation.do_backward_forward_and_message(
        A_matrix, p0, p1, num_time_steps, obs_messages, num_users,
        map_backward_message, map_forward_message, (0, num_users),
        clip_lower=0.0, clip_upper=clip_upper, quantization=quantization))

  # Collect beliefs
  bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)

  np.testing.assert_array_almost_equal(post_exp_bp.shape, post_exp_bp.shape)
  np.testing.assert_array_almost_equal(bp_beliefs, post_exp_bp, decimal=2)


def test_bp_wrapped_rdp():
  (num_users, num_time_steps, p0, p1, _, _, _,
    _, q_marginal_infected, observations_all, contacts_all,
    g_param, h_param, alpha, beta) = prepare_example()

  q_user0 = 0.6
  q_marginal_infected[0] = q_user0

  # This is a stochastic test. Increase the number of samples for less variance
  num_samples = 30
  results_pybind = np.zeros((num_samples))
  results_numba = np.zeros((num_samples))

  # Numba preparation
  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta], dtype=np.float32),
    1: np.array([beta, beta, 1-alpha, beta], dtype=np.float32),
  }

  obs_messages = np.ones((num_users, num_time_steps, 4), dtype=np.float32)
  for obs in observations_all:
    obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

  map_forward_message, map_backward_message = (
    belief_propagation.init_message_maps(
      contacts_all, (0, num_users)))
  np.testing.assert_array_almost_equal(
    map_forward_message[1][0], [0., 1., 1., 0.])
  map_forward_message[1][0][3] = q_user0

  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-g_param, g_param, 0],
    [0, 0, 1-h_param, h_param],
    [0, 0, 0, 1]
  ], dtype=np.float32)

  user_test = 1
  for num_sample in range(num_samples):
    np.testing.assert_array_almost_equal(
      map_forward_message[1][0], [0., 1., 1., q_user0])
    # Numba case
    bp_noised, _, _ = (
      belief_propagation.forward_backward_user(
        A_matrix, p0, p1, user_test, map_backward_message[user_test],
        map_forward_message[user_test], num_time_steps, obs_messages[user_test],
        -1., .6, epsilon_dp=10.1, a_rdp=8.7))
    results_numba[num_sample] = bp_noised[3][2]

  # Pybind case
    post_exp_out = dpfn_util.forward_backward_user_wrapped(
      num_user=user_test,
      num_users=num_users,
      num_time_steps=num_time_steps,
      probab0=p0,
      probab1=p1,
      g_param=g_param,
      h_param=h_param,
      alpha=alpha,
      beta=beta,
      rho_rdp=10.1,  # rho=0.11, a=8.7 corresponds to (e,d)(1., 0.001)
      a_rdp=8.7,
      clip_lower=-1.,
      clip_upper=0.6,
      p_infected=q_marginal_infected,
      observations=observations_all,
      contacts=contacts_all)
    results_pybind[num_sample] = post_exp_out[1][3][2]

  mean_pybind = np.mean(results_pybind)
  mean_numba = np.mean(results_numba)
  std_pybind = np.std(results_pybind)
  std_numba = np.std(results_numba)

  print(f"Means pybind: {mean_pybind:6.3f}  -  Numba: {mean_numba:6.3f}")
  print(f"STD pybind: {std_pybind:6.3f}  -  Numba: {std_numba:6.3f}")

  assert np.abs(mean_pybind - mean_numba) < 0.02, (
    f"Means of pybind {mean_pybind:.3f} and Numba "
    f"{mean_numba:.3f} don't match")
  assert np.abs(std_pybind - std_numba) < 0.02, (
    f"STD of pybind {std_pybind:.3f} and Numba "
    f"{std_numba:.3f} don't match")
