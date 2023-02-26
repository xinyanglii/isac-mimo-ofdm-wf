import sys
from os.path import abspath, dirname

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import numpy as np
import torch
from hydra import compose, initialize

from src.stochastic_manopt import ComplexSphere, StochasticREPMSConjugateGradient


def test_manopt():
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="test_config")
    m, n = cfg.num_ant_tx, cfg.num_streams
    manifold = ComplexSphere(m, n)
    random_point = torch.zeros((m, n), dtype=torch.cfloat)
    random_point[:, 0] = 1
    random_point = random_point / torch.linalg.norm(random_point)

    random_perturb_dist = torch.distributions.Normal(loc=0, scale=1e-1)

    def cost(point, param_list=None):
        return torch.linalg.norm(point - random_point) ** 2

    def stoch_cost(point, param_list):
        pertrubs = param_list[0]
        perturb_point = random_point + pertrubs.mean() * (torch.ones((m, n)) + 1j * torch.ones((m, n)))
        perturb_point = perturb_point / torch.linalg.norm(perturb_point)
        return torch.linalg.norm(point - perturb_point) ** 2

    # unconstrained case
    uncons_optimizer = StochasticREPMSConjugateGradient()
    uncons_result = uncons_optimizer.run(manifold, cost)
    assert np.allclose(uncons_result["manopt_result"].point, random_point, atol=1e-6)
    assert (
        len(uncons_result["objval_convergence"])
        == len(uncons_result["max_ineqval_convergence"])
        == len(uncons_result["gradient_norm_convergence"])
        == uncons_result["manopt_result"].iterations
    )

    # constrained case
    cons_optimizer = StochasticREPMSConjugateGradient(violation_tolerance=1e-3)

    def ineq_constraint(point, param_list=None):
        return torch.linalg.norm(point, dim=0)[: n // 2] - 0.05 / m

    cons_result = cons_optimizer.run(manifold, cost, ineq_constraint=ineq_constraint)
    assert (
        ineq_constraint(torch.from_numpy(cons_result["manopt_result"].point)).max()
        <= cons_optimizer.violation_tolerance
    )
    assert (
        cons_result["manopt_result"].iterations
        == len(cons_result["penalty_factors"])
        == len(cons_result["smoothing_factors"])
    )

    # stochastic unconstrained case
    stoch_uncons_optimizer = StochasticREPMSConjugateGradient(num_points_per_iter=10)
    stoch_uncons_result = stoch_uncons_optimizer.run(manifold, stoch_cost, param_sampler_list=[random_perturb_dist])
    assert np.allclose(stoch_uncons_result["manopt_result"].point, random_point, atol=random_perturb_dist.scale.item())
    assert stoch_uncons_result["manopt_result"].iterations > 500

    # stochastic objective + deterministic ineq
    stoch_determ_optimizer = StochasticREPMSConjugateGradient(num_points_per_iter=10, violation_tolerance=1e-3)
    stoch_determ_result = stoch_determ_optimizer.run(
        manifold,
        stoch_cost,
        param_sampler_list=[random_perturb_dist],
        ineq_constraint=ineq_constraint,
    )
    assert (
        ineq_constraint(torch.from_numpy(stoch_determ_result["manopt_result"].point)).max()
        <= stoch_determ_optimizer.violation_tolerance
    )
