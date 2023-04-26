import time
from copy import deepcopy

import numpy as np
import pymanopt as pm
import torch
from pymanopt.manifolds.manifold import RiemannianSubmanifold
from pymanopt.optimizers import ConjugateGradient
from pymanopt.tools import printer


class _ComplexSphereBase(RiemannianSubmanifold):
    def __init__(self, *shape, name, dimension):
        if len(shape) == 0:
            raise TypeError("Need at least one dimension.")
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.pi

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a.conj(),
            tangent_vector_b,
            axes=tangent_vector_a.ndim,
        ).real

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        # inner = max(min(self.inner_product(point_a, point_a, point_b), 1), -1)
        tmp = 2 * np.arcsin(0.5 * np.linalg.norm(point_a - point_b))
        return tmp.real

    def projection(self, point, vector):
        return vector - self.inner_product(point, point, vector) * point

    to_tangent_space = projection

    def weingarten(self, point, tangent_vector, normal_vector):
        return -self.inner_product(point, point, normal_vector) * tangent_vector

    def exp(self, point, tangent_vector):
        norm = self.norm(point, tangent_vector)
        return point * np.cos(norm) + tangent_vector * np.sinc(norm / np.pi)

    def retraction(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.projection(point_a, point_b - point_a)
        distance = self.dist(point_a, point_b)
        epsilon = np.finfo(np.float64).eps
        factor = (distance + epsilon) / (self.norm(point_a, vector) + epsilon)
        return factor * vector

    def random_point(self):
        point = np.random.normal(size=self._shape) + 1j * np.random.normal(size=self._shape)
        return self._normalize(point)

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=self._shape) + 1j * np.random.normal(size=self._shape)
        return self._normalize(self.projection(point, vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def pair_mean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zero_vector(self, point):
        return np.zeros(self._shape, dtype=complex)

    def _normalize(self, array):
        return array / np.linalg.norm(array)


class ComplexSphere(_ComplexSphereBase):
    r"""The sphere manifold.

    Manifold of shape :math:`n_1 \times \ldots \times n_k` tensors with unit
    Euclidean norm.
    The norm is understood as the :math:`\ell_2`-norm of :math:`\E =
    \R^{\sum_{i=1}^k n_i}` after identifying :math:`\R^{n_1 \times \ldots
    \times n_k}` with :math:`\E`.
    The metric is the one inherited from the usual Euclidean inner product that
    induces :math:`\norm{\cdot}_2` on :math:`\E` such that the manifold forms a
    Riemannian submanifold of Euclidean space.

    Args:
        shape: The shape of tensors.
    """

    def __init__(self, *shape: int):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            (n,) = shape
            name = f"Sphere manifold of {n}-vectors"
        elif len(shape) == 2:
            m, n = shape
            name = f"Sphere manifold of {m}x{n} matrices"
        else:
            name = f"Sphere manifold of shape {shape} tensors"
        dimension = np.prod(shape) - 1
        super().__init__(*shape, name=name, dimension=dimension)


class StochasticREPMSConjugateGradient(ConjugateGradient):
    def __init__(
        self,
        beta_rule: str = "HestenesStiefel",
        orth_value=np.inf,
        line_searcher=None,
        num_points_per_iter=10,
        initial_penalty=1.0,
        penalty_update=2,
        max_penalty=2**20,
        initial_smooth_factor=0.1,
        min_smooth_factor=1e-6,
        smooth_factor_update=None,
        violation_tolerance=1e-2,
        *args,
        **kwargs,
    ):
        r"""Stochastic Riemmanian Exact Penalty Method via Smoothing, implemented based on pymanopt
        Liu, Changshuo, and Nicolas Boumal. "Simple algorithms for optimization on Riemannian manifolds with
        constraints." Applied Mathematics & Optimization 82 (2020): 949-981.
        Li, Xinyang, et al. "Optimal and Robust Waveform Design for MIMO-OFDM Channel Sensing:
        A Cram\'er-Rao Bound Perspective." arXiv preprint arXiv:2301.10689 (2023).

        :param num_points_per_iter: Number of points to sample at each iteration. Defaults to 10.
        :type num_points_per_iter: int, optional
        :param initial_penalty: initial penalty factor for inequality constraints. Defaults to 1..
        :type initial_penalty: float, optional
        :param penalty_update: update factor of penalty at each iteration. Defaults to 2.
        :type penalty_update: float, optional
        :param max_penalty: manximum penalty factor, stop updating penalty if reached. Defaults to 2**20.
        :type max_penalty: float, optional
        :param initial_smooth_factor: initial smoothing factor for inequality constraints. Defaults to 0.1.
        :type initial_smooth_factor: float, optional
        :param min_smooth_factor: minimum smoothing factor, stop updating if reached. Defaults to 1e-6.
        :type min_smooth_factor: float, optional
        :param smooth_factor_update: updating factor for smoothing factor. Defaults to (u_min / u0)**(1/30)
            where u0, u_min are initial smoothing factor and minimum smoothing factor respectively.
        :type smooth_factor_update: float | None, optional
        :param violation_tolerance: tolerance of constraints violation, stop updating penalty and smoothing
            factor if constraint value below this threshold. Defaults to 1e-2.
        :type violation_tolerance: float, optional
        """
        super().__init__(beta_rule, orth_value, line_searcher, *args, **kwargs)

        self.num_points_per_iter = num_points_per_iter

        self.penalty0 = initial_penalty
        self.penalty_update = penalty_update
        self.max_penalty = max_penalty

        self.smooth0 = initial_smooth_factor
        self.min_smooth = min_smooth_factor
        if not smooth_factor_update:
            self.smooth_update = (self.min_smooth / self.smooth0) ** (1 / 30)
        else:
            self.smooth_update = smooth_factor_update

        self.violation_tolerance = violation_tolerance

    # TODO: implement equality constrained optimization
    def run(
        self,
        manifold,
        base_problem,
        *,
        param_sampler_list=[],
        ineq_constraint=None,
        initial_point=None,
        reuse_line_searcher=False,
    ):
        r"""solver implementation of SREPMS

        :param manifold: pymanopt manifold, on which variables is to be optimized
        :type manifold: pymanopt.manifolds.manifold.Manifold
        :param base_problem: handle objective function, in the form of fun(x, params_list), x the variable (pytorch
            Tensor) to be optimized on the manifold, params_list the list of random parameters sampled by
            param_sampler_list (see below) during optimization.
        :type base_problem: Callable
        :param param_sampler_list: list of parameter samplers, each can be called as param_sampler.sample(size) to
            sample parameters, recommended as torch.distributions object. If set to [], problem becomes deterministic.
        :type param_sampler_list: list, optional
        :param ineq_constraint: function that computes inequality constraint values, in the form fun(x, params_list),
            x the variable (pytorch Tensor) to be optimized on the manifold, params_list the list of random
            parameters sampled by params_list (see above) during optimization. If multiple inequality constraints
            are used, return a 1D torch Tensor of constraint values. Defaults to None.
        :type ineq_constraint: Callable | None, optional
        :param initial_point: initial point for optimization, if None, use random point on manifold. Defaults to None.
        :type initial_point: torch.Tensor | None, optional
        :param reuse_line_searcher: whether to reuse line searcher.
        :type reuse_line_searcher: bool, optional
        :return: solution of optimization problem
        :rtype: pymanopt.optimizers.optimizer.OptimizerResult
        """
        alpha = self.penalty0 if ineq_constraint else 0
        u = self.smooth0
        penalty_list = [alpha]
        smoothing_list = [u]
        objval_convergence = []
        max_ineqval_convergence = []
        gradient_norm_convergence = []

        params_list = [sampler.sample((self.num_points_per_iter,)) for sampler in param_sampler_list]
        problem = self.update_problem(manifold, base_problem, params_list, alpha, u, ineq_constraint)

        manifold = problem.manifold
        cost_fun = problem.cost
        gradient = problem.riemannian_gradient

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        line_searcher = self.line_searcher

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ],
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        # Calculate initial cost-related quantities.
        cost = cost_fun(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x, grad)
        Pgrad = problem.preconditioner(x, grad)
        gradPgrad = manifold.inner_product(x, grad, Pgrad)

        # record problem convergence
        objval = base_problem(torch.from_numpy(x), params_list).numpy(force=True)
        objval_convergence.append(objval)
        max_ineqval = (
            ineq_constraint(torch.from_numpy(x), params_list).max().numpy(force=True) if ineq_constraint else 0
        )
        max_ineqval_convergence.append(max_ineqval)
        gradient_norm_convergence.append(gradient_norm)

        # Initial descent direction is the negative gradient.
        descent_direction = -Pgrad

        self._initialize_log(
            optimizer_parameters={
                "beta_rule": self._beta_rule,
                "orth_value": self._orth_value,
                "line_searcher": line_searcher,
            },
        )

        # Initialize iteration counter and timer.
        iteration = 0
        step_size = np.nan
        start_time = time.time()

        while True:
            iteration += 1

            column_printer.print_row([iteration, cost, gradient_norm])

            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
            )

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                gradient_norm=gradient_norm,
                iteration=iteration,
                step_size=step_size,
            )

            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = manifold.inner_product(x, grad, descent_direction)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if self._verbosity >= 3:
                    print(
                        "Conjugate gradient info: got an ascent direction "
                        f"(df0 = {df0:.2f}), reset to the (preconditioned) "
                        "steepest descent direction.",
                    )
                # Reset to negative gradient: this discards the CG memory.
                descent_direction = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            step_size, newx = line_searcher.search(
                cost_fun,
                manifold,
                x,
                descent_direction,
                cost,
                df0,
            )

            # Compute the new cost-related quantities for newx
            # newcost = objective(newx)
            newgrad = gradient(newx)
            # newgradient_norm = manifold.norm(newx, newgrad)
            Pnewgrad = problem.preconditioner(newx, newgrad)
            newgradPnewgrad = manifold.inner_product(newx, newgrad, Pnewgrad)

            # Powell's restart strategy.
            oldgrad = manifold.transport(x, newx, grad)
            orth_grads = manifold.inner_product(newx, oldgrad, Pnewgrad) / newgradPnewgrad
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                descent_direction = -Pnewgrad
            else:
                # Transport latest search direction to tangent space at new
                # estimate.
                descent_direction = manifold.transport(
                    x,
                    newx,
                    descent_direction,
                )
                beta = self._beta_update(
                    manifold=manifold,
                    x=x,
                    newx=newx,
                    grad=grad,
                    newgrad=newgrad,
                    Pnewgrad=Pnewgrad,
                    newgradPnewgrad=newgradPnewgrad,
                    Pgrad=Pgrad,
                    gradPgrad=gradPgrad,
                    gradient_norm=gradient_norm,
                    oldgrad=oldgrad,
                    descent_direction=descent_direction,
                )
                descent_direction = -Pnewgrad + beta * descent_direction

            x = newx

            if (
                ineq_constraint
                and alpha
                and (ineq_constraint(torch.from_numpy(x), params_list) > self.violation_tolerance).any()
            ):
                alpha = min(self.penalty_update * alpha, self.max_penalty)
                u = max(self.min_smooth, self.smooth_update * u)

            penalty_list.append(alpha)
            smoothing_list.append(u)

            params_list = [sampler.sample((self.num_points_per_iter,)) for sampler in param_sampler_list]
            problem = self.update_problem(manifold, base_problem, params_list, alpha, u, ineq_constraint)

            manifold = problem.manifold
            cost_fun = problem.cost
            gradient = problem.riemannian_gradient

            cost = cost_fun(newx)
            grad = gradient(newx)
            gradient_norm = manifold.norm(newx, grad)
            Pgrad = problem.preconditioner(newx, grad)
            gradPgrad = manifold.inner_product(newx, grad, Pgrad)

            # record problem convergence
            objval = base_problem(torch.from_numpy(x), params_list).numpy(force=True)
            objval_convergence.append(objval)
            max_ineqval = (
                ineq_constraint(torch.from_numpy(x), params_list).max().numpy(force=True) if ineq_constraint else 0
            )
            max_ineqval_convergence.append(max_ineqval)
            gradient_norm_convergence.append(gradient_norm)

        result = self._return_result(
            start_time=start_time,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )

        return {
            "manopt_result": result,
            "objval_convergence": np.array(objval_convergence),
            "max_ineqval_convergence": np.array(max_ineqval_convergence),
            "gradient_norm_convergence": np.array(gradient_norm_convergence),
            "penalty_factors": np.array(penalty_list),
            "smoothing_factors": np.array(smoothing_list),
        }

    def update_problem(self, manifold, base_problem, params_list, alpha, u, ineq_constraint):
        @pm.function.pytorch(manifold)
        def cost(X):
            if alpha:
                ineq_val = ineq_constraint(X, params_list)
                ineq_val[ineq_val < 0] = 0
                ineq_val[(0 < ineq_val) & (ineq_val <= u)] = ineq_val[(0 < ineq_val) & (ineq_val <= u)] ** 2 / (2 * u)
                ineq_val[ineq_val > u] = ineq_val[ineq_val > u] - u / 2
                penalty = alpha * ineq_val.sum()
            else:
                penalty = 0
            return base_problem(X, params_list) + penalty

        problem = pm.Problem(manifold, cost)
        return problem
