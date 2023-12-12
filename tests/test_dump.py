"""Test functions for the data dump."""
import numpy as np
import dpfn_util  # C++ implementation
import os


def test_fn_full():
  # Load test case
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

  # Run the dump function
  q_marginal = np.random.rand(num_users, num_time_steps).astype(np.float32)
  datadump = dpfn_util.fn_features_dump(
    num_workers=1,
    num_users=num_users,
    num_time_steps=num_time_steps,
    q_marginal=q_marginal,
    contacts=contacts_all,
    users_age=9*np.ones(num_users, dtype=np.int32))

  # Check the first entry, which is a contact from user 1 on day 1
  expected = np.array([1, 1, 9, int(q_marginal[1][1]*1024)])
  np.testing.assert_array_almost_equal(datadump[0][0], expected, decimal=5)

  assert np.all(datadump[:, :, 3] <= 1024)

  # Try to dump the dataset
  dirname = '/tmp/test_dump'
  os.makedirs(dirname, exist_ok=True)

  for user in range(num_users):
    fname = os.path.join(dirname, f'user_{user:50d}.npz')
    np.savez(
      fname,
      contacts=datadump[user],
      fn_pred=q_marginal[user][-1],
      sim_state=np.random.randint(low=0, high=4))
