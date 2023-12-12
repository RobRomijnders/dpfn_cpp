"""Test against the original dpfn library."""
import numpy as np
from dpfn import constants, inference, util
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
  alpha, beta = 0.0001, 0.001

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


def test_d_terms():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8
  p0, p1 = 0.01, 0.3

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.7, .7, .7, .7, .7, .7, .7, .7],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ], dtype=np.float32)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term_new, d_no_term_new = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=p0,
    p1=p1,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  d_terms = dpfn_util.d_terms_wrapped(
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

  np.testing.assert_array_almost_equal(d_terms[0], d_term_new)
  np.testing.assert_array_almost_equal(d_terms[1], d_no_term_new)


def test_d_terms_repeats():
  contacts_all = np.array([
    (2, 1, 4, 1),
    (2, 1, 1, 1),
    (1, 1, 1, 1),
    (2, 1, 2, 1),
    ], dtype=np.int32)
  num_time_steps = 8
  p0, p1 = 0.1, 0.3

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.6, .6, .6, .6, .8, .8, .8, .8],
  ], dtype=np.float32)

  # No repeats
  d_terms_before = np.array(dpfn_util.d_terms_wrapped(
    num_user=user,
    num_time_steps=num_time_steps,
    q_marginal=q_marginal_infected,
    contacts=contacts_all,
    p0=p0,
    p1=p1,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,))

  # Remove repeats
  d_terms_after = np.array(dpfn_util.d_terms_wrapped(
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
    dedup_contacts=1))

  np.testing.assert_array_almost_equal(
    d_terms_before[1][2], np.log(1. - p1*0.6) + np.log(1. - p1*0.1))

  # After deduplication, only one contact happened on day 1
  np.testing.assert_array_almost_equal(
    d_terms_after[1][2], np.log(1. - p1*0.1))

  # After deduplication, no contacts happened on day 0
  np.testing.assert_array_almost_equal(
    d_terms_after[1][3], 0.0)

  # Independent of dedup, from day 3 onwards, the dterms should be the same
  np.testing.assert_array_almost_equal(
    d_terms_before[:, 4:], d_terms_after[:, 4:])


def test_fn_wrapped():
  (num_users, num_time_steps, p0, p1, seq_array_hot, log_c_z_u, log_A_start,
    past_contacts, q_marginal_infected, _, contacts_all,
    _, _, _, _) = prepare_example()

  post_exp_out = dpfn_util.fn_wrapped_func(
    user_start=0,
    user_end=num_users,
    num_time_steps=num_time_steps,
    probab0=p0,
    probab1=p1,
    rho_rdp=-1.,
    a_rdp=-1.,
    clip_lower=-1.,
    clip_upper=10.,
    seq_array_hot=seq_array_hot.astype(np.int32),
    log_c_z_u=log_c_z_u,
    log_a_start=log_A_start,
    p_infected_matrix=q_marginal_infected,
    contacts=contacts_all)

  post_exp_py, _, _ = inference.fn_step_wrapped(
    (0, num_users),
    seq_array_hot,
    log_c_z_u,  # already depends in mpi_rank
    log_A_start,
    q_marginal_infected,
    num_time_steps,
    probab0=p0,
    probab1=p1,
    clip_lower=-1.,
    clip_upper=10000.,
    past_contacts_array=past_contacts,
    dp_method=-1,
    epsilon_dp=-1.,
    delta_dp=-1.,
    quantization=-1)

  np.testing.assert_array_almost_equal(post_exp_out.shape, post_exp_py.shape)
  np.testing.assert_array_almost_equal(post_exp_out, post_exp_py, decimal=5)


def test_fn_wrapped_rdp():
  (num_users, num_time_steps, p0, p1, seq_array_hot, log_c_z_u, log_A_start,
    past_contacts, q_marginal_infected, _, contacts_all,
    _, _, _, _) = prepare_example()

  # This is a stochastic test. Increase the number of samples for less variance
  num_samples = 30
  results_pybind = np.zeros((num_samples))
  results_numba = np.zeros((num_samples))

  for num_sample in range(num_samples):
    post_exp_out = dpfn_util.fn_wrapped_func(
      user_start=0,
      user_end=num_users,
      num_time_steps=num_time_steps,
      probab0=p0,
      probab1=p1,
      rho_rdp=.5,
      a_rdp=4.,
      clip_lower=0.001,
      clip_upper=0.99,
      seq_array_hot=seq_array_hot.astype(np.int32),
      log_c_z_u=log_c_z_u,
      log_a_start=log_A_start,
      p_infected_matrix=q_marginal_infected,
      contacts=contacts_all)
    results_pybind[num_sample] = post_exp_out[1][3][2]

    post_exp_py, _, _ = inference.fn_step_wrapped(
      (0, num_users),
      seq_array_hot,
      log_c_z_u,  # already depends in mpi_rank
      log_A_start,
      q_marginal_infected,
      num_time_steps,
      probab0=p0,
      probab1=p1,
      clip_lower=0.001,
      clip_upper=0.99,
      past_contacts_array=past_contacts,
      dp_method=5,
      epsilon_dp=0.5,
      a_rdp=4.,
      quantization=-1)
    results_numba[num_sample] = post_exp_py[1][3][2]

  mean_pybind = np.mean(results_pybind)
  mean_numba = np.mean(results_numba)
  assert np.abs(mean_pybind - mean_numba) < 0.01, (
    f"Means of pybind {mean_pybind:.3f} and numba {mean_numba:.3f} don't match")

  std_pybind = np.std(results_pybind)
  std_numba = np.std(results_numba)
  assert np.abs(std_pybind - std_numba) < 0.01, (
    f"STDs of pybind {std_pybind:.3f} and numba {std_numba:.3f} don't match")


def test_fn_full():
  (num_users, num_time_steps, p0, p1, _, _, _,
    _, _, observations_all, contacts_all,
    g_param, h_param, alpha, beta) = prepare_example()

  post_exp_out, _ = dpfn_util.fn_full_func(
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
    clip_upper=10.,
    quantization=-1,
    observations=observations_all,
    contacts=contacts_all,
    users_age=-1*np.ones(num_users, dtype=np.int32))

  post_exp_py = inference.fact_neigh(
    num_users,
    num_time_steps,
    observations_all,
    contacts_all,
    probab_0=p0,
    probab_1=p1,
    alpha=alpha,
    beta=beta,
    g_param=g_param,
    h_param=h_param,
    clip_lower=-1.,
    clip_upper=10000.)

  # with np.printoptions(precision=2, suppress=True):
  #   print(post_exp_out[0][:3])

  np.testing.assert_array_almost_equal(post_exp_py.shape, post_exp_out.shape)
  np.testing.assert_array_almost_equal(post_exp_py, post_exp_out, decimal=3)


def test_fn_full_rdp():
  (num_users, num_time_steps, p0, p1, _, _, _,
    _, _, observations_all, contacts_all,
    g_param, h_param, alpha, beta) = prepare_example()

  post_exp_rdp, _ = dpfn_util.fn_full_func(
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
    rho_rdp=1000.,
    a_rdp=1.1,
    clip_lower=0.01,
    clip_upper=0.5,
    quantization=128,
    observations=observations_all,
    contacts=contacts_all,
    users_age=-1*np.ones(num_users, dtype=np.int32))

  # With lognormal noise, the rdp values are unbiased, so keep zero noise here
  post_exp_py = inference.fact_neigh(
    num_users,
    num_time_steps,
    observations_all,
    contacts_all,
    probab_0=p0,
    probab_1=p1,
    alpha=alpha,
    beta=beta,
    g_param=g_param,
    h_param=h_param,
    clip_lower=0.01,
    clip_upper=0.5,
    quantization=128)

  np.testing.assert_array_almost_equal(post_exp_py.shape, post_exp_rdp.shape)
  # This test is under RDP, only check that the values are in the correct range
  assert np.mean(np.abs(post_exp_py - post_exp_rdp)) < 0.02, "Stochastic test!"


def test_contact_flip():
  num_users = 3
  num_msg = constants.CTC
  contacts_all = np.array([
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ], dtype=np.int32)

  past_contacts = dpfn_util.contact_flip(
    num_users=num_users, contacts=contacts_all)
  past_contacts = np.reshape(past_contacts, [num_users, num_msg, 2])

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, num_msg, 2])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1])

  np.testing.assert_equal(actual=past_contacts.dtype, desired=np.int32)


def test_past_contacts_contact_flip():
  num_users = 3
  num_msg = constants.CTC
  contacts_all = np.array([
    (1, 2, 4, 1),
    (1, 2, 3, 1),
    (1, 2, 2, 1),
    (1, 2, 1, 1),
    (2, 1, 4, 1),
    ], dtype=np.int32)

  past_contacts = dpfn_util.contact_flip(
    num_users=num_users, contacts=contacts_all).reshape([num_users, num_msg, 2])

  past_contacts_static, _ = util.get_past_contacts_static(
    (0, 3), contacts_all, num_msg=num_msg)

  # Silly to test set, but contacts could occur in any order ofcourse
  # Check that the values match
  np.testing.assert_equal(
    actual=set(past_contacts_static[0].flatten().tolist()),
    desired=set(past_contacts[0].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[1].flatten().tolist()),
    set(past_contacts[1].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[2].flatten().tolist()),
    set(past_contacts[2].flatten().tolist())
  )


def test_contacts_quantiles():
  num_users = 10
  num_time_steps = 5
  p0, p1 = 0.01, 0.3
  g_param, h_param = 0.2, 0.2
  alpha, beta = 0.0001, 0.001

  observations_all = np.array([
    (0, 2, 1)
  ], dtype=np.int32)

  contacts_all = np.array([
    (1, 0, 1, 1),
    (2, 0, 1, 1),
    (3, 0, 1, 1),
    (5, 4, 2, 1),
    (6, 4, 2, 1),
    (7, 4, 2, 1),
    (8, 4, 2, 1),
    (9, 4, 2, 1),
  ], dtype=np.int32)

  users_age = np.array([
    0,  # 0
    3,  # 1
    5,  # 2
    8,  # 3
    4,  # 4
    1,  # 5
    2,  # 6
    5,  # 7
    7,  # 8
    8,  # 9
  ], dtype=np.int32)

  _, contacts_age = dpfn_util.fn_full_func(
    num_workers=1,
    num_rounds=1,
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
    quantization=-1,
    observations=observations_all,
    contacts=contacts_all,
    users_age=users_age)

  np.testing.assert_equal(contacts_age.shape, (2*num_users, ))
  contacts_age = contacts_age.reshape([2, num_users])

  np.testing.assert_array_almost_equal(contacts_age[0][0], 5)  # quantile 50
  np.testing.assert_array_almost_equal(contacts_age[1][0], 8)  # quantile 80
  np.testing.assert_array_almost_equal(contacts_age[0][4], 5)  # quantile 50
  np.testing.assert_array_almost_equal(contacts_age[1][4], 7)  # quantile 80
