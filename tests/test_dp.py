import math

import numpy as np

from fedmed.privacy.dp import RDPAccountant, add_gaussian_noise, clip_update


def test_clip_bounds_norm():
    delta = np.ones(100, dtype=np.float32) * 3
    clipped, norm, was = clip_update(delta, 1.0)
    assert was and math.isclose(norm, 30.0, rel_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(clipped)), 1.0, rel_tol=1e-5)
    small, _, was = clip_update(np.full(4, 0.1, dtype=np.float32), 1.0)
    assert not was and np.allclose(small, 0.1)


def test_noise_scale():
    noise = add_gaussian_noise(np.zeros(200_000, dtype=np.float32), clip_norm=2.0, noise_multiplier=0.5,
                               rng=np.random.default_rng(0))
    assert abs(noise.std() - 1.0) < 0.02


def test_accountant_monotone_and_closed_form():
    acc = RDPAccountant(noise_multiplier=1.0, delta=1e-5)
    assert acc.epsilon() == 0.0
    eps = [acc.epsilon(t) for t in (1, 2, 4, 8)]
    assert eps == sorted(eps)
    # Gaussian RDP optimum: eps = T/(2 s^2) + 2*sqrt(T ln(1/d) / (2 s^2)); compare to numerical search.
    t, s, d = 8, 1.0, 1e-5
    closed = t / (2 * s * s) + math.sqrt(2 * t * math.log(1 / d)) / s
    assert acc.epsilon(t) <= closed * 1.02
    assert acc.epsilon(t) >= closed * 0.8


def test_more_noise_means_smaller_epsilon_and_no_noise_is_infinite():
    assert RDPAccountant(2.0).epsilon(8) < RDPAccountant(1.0).epsilon(8)
    assert RDPAccountant(0.0).epsilon(1) == math.inf
