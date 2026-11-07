"""
Inverse Material Parameter Identification and Constitutive Calibration.

Solves the inverse nonlinear least-squares optimization problem:
    min_{theta} 1/2 * sum_{j=1}^M w_j (F_sim(delta_j; theta) - F_exp(delta_j))^2 + beta / 2 * ||theta - theta_0||^2

where:
    - theta in R^P is the unknown constitutive parameter vector (e.g. [E], [E, nu], [C_10], [C_10, K_bulk]).
    - delta_j is the prescribed displacement or indentation depth at step j.
    - F_exp(delta_j) is the experimentally acquired reaction force or indenter load.
    - F_sim(delta_j; theta) is the model-simulated structural force response.
    - w_j > 0 is the data point confidence weight.
    - beta >= 0 is the Tikhonov prior regularization weight penalizing deviation from initial guess theta_0.

Algorithms Supported:
---------------------
- Trust-Region Reflective (TRF): Bound-constrained nonlinear least squares (default).
- Dogleg / Levenberg-Marquardt (LM): Unconstrained Gauss-Newton least squares.
- L-BFGS-B: Bound-constrained quasi-Newton optimization.

References:
-----------
- A. Tarantola (2005). "Inverse Problem Theory and Methods for Model Parameter
  Estimation". SIAM.
- J. Nocedal & S.J. Wright (2006). "Numerical Optimization". Springer (Ch. 10,
  Nonlinear Least-Squares Problems).
- J. Bonet & R.D. Wood (2008). "Nonlinear Continuum Mechanics for Finite
  Element Analysis". Cambridge University Press.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import scipy.optimize as spopt


@dataclass(frozen=True)
class ParameterBounds:
    """
    Specification for a single calibratable constitutive parameter.

    Attributes:
    -----------
    name : str
        Human-readable parameter name (e.g. 'youngs_modulus', 'c10', 'poissons_ratio').
    initial_value : float
        Initial guess theta_0 for optimization.
    lower_bound : float, default=0.0
        Minimum allowable physical value.
    upper_bound : float, default=np.inf
        Maximum allowable physical value.
    description : str, optional
        Physical unit and description.
    """

    name: str
    initial_value: float
    lower_bound: float = 0.0
    upper_bound: float = np.inf
    description: str = ""

    def __post_init__(self) -> None:
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be strictly less than "
                f"upper_bound ({self.upper_bound}) for parameter '{self.name}'."
            )
        if not (self.lower_bound <= self.initial_value <= self.upper_bound):
            raise ValueError(
                f"initial_value ({self.initial_value}) must lie within "
                f"[{self.lower_bound}, {self.upper_bound}] for parameter '{self.name}'."
            )


@dataclass(frozen=True)
class IdentificationResult:
    """
    Container for material parameter identification and calibration results.

    Attributes:
    -----------
    parameters : dict[str, float]
        Calibrated optimal parameter mapping {name: optimal_value}.
    initial_parameters : dict[str, float]
        Initial nominal parameter mapping {name: initial_value}.
    simulated_forces : np.ndarray
        Model force response evaluated at optimal parameters theta*.
    experimental_forces : np.ndarray
        Target experimental force observations F_exp.
    residuals : np.ndarray
        Force discrepancy F_sim - F_exp.
    rms_error : float
        Root-mean-square force discrepancy sqrt(mean(residual^2)).
    relative_error : float
        Relative force error ||F_sim - F_exp|| / ||F_exp||.
    iterations : int
        Number of optimizer iterations performed.
    n_evaluations : int
        Number of forward simulation model evaluations.
    cost : float
        Final objective value 0.5 * sum(residuals^2).
    success : bool
        Whether optimization converged successfully.
    message : str
        Convergence diagnostic message.
    wall_time_seconds : float
        Total calibration wall-clock time in seconds.
    parameter_history : list[dict[str, float]]
        Parameter trajectories across iterations (if tracked).
    """

    parameters: dict[str, float]
    initial_parameters: dict[str, float]
    simulated_forces: np.ndarray
    experimental_forces: np.ndarray
    residuals: np.ndarray
    rms_error: float
    relative_error: float
    iterations: int
    n_evaluations: int
    cost: float
    success: bool
    message: str
    wall_time_seconds: float
    parameter_history: list[dict[str, float]] = field(default_factory=list)


class MaterialParameterIdentifier:
    """
    Nonlinear Least-Squares Optimizer for Material Parameter Identification.

    Calibrates constitutive material parameters theta by minimizing force
    residuals between forward FEM simulations and experimental data:
        min_{theta} 1/2 * ||r(theta)||^2 + beta / 2 * ||theta - theta_0||^2
    """

    def __init__(
        self,
        parameters: Sequence[ParameterBounds] | Sequence[str] | dict[str, tuple[float, float, float]],
        beta: float = 0.0,
        weights: np.ndarray | Sequence[float] | None = None,
        method: str = "trf",
        ftol: float = 1e-4,
        xtol: float = 1e-4,
        gtol: float = 1e-4,
        max_nfev: int | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        parameters : Sequence of ParameterBounds, or dict {name: (initial, lower, upper)}
            Calibratable parameter specifications.
        beta : float, default=0.0
            Tikhonov prior regularization weight penalizing deviation from initial guess (beta >= 0).
        weights : Sequence of float, optional
            Observation channel weights w_j > 0 for each force data point.
        method : {'trf', 'dogbox', 'lm', 'L-BFGS-B'}, default='trf'
            Optimization method:
            - 'trf': Trust Region Reflective (handles bounds robustly).
            - 'dogbox': Dogleg with rectangular trust regions.
            - 'lm': Levenberg-Marquardt (unbounded problems).
            - 'L-BFGS-B': Bound-constrained quasi-Newton.
        ftol, xtol, gtol : float, default=1e-4
            Convergence tolerances.
        max_nfev : int, optional
            Maximum allowed forward model evaluations.
        """
        if beta < 0.0:
            raise ValueError(f"Tikhonov regularization beta must be non-negative, got {beta}.")

        self.beta = float(beta)
        self.weights = weights
        self.method = method
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.gtol = float(gtol)
        self.max_nfev = max_nfev

        self._param_bounds = self._resolve_parameter_bounds(parameters)
        self._param_names = tuple(p.name for p in self._param_bounds)
        self._x0 = np.array([p.initial_value for p in self._param_bounds], dtype=np.float64)
        self._lower_bounds = np.array([p.lower_bound for p in self._param_bounds], dtype=np.float64)
        self._upper_bounds = np.array([p.upper_bound for p in self._param_bounds], dtype=np.float64)

    @classmethod
    def from_bounds(
        cls,
        bounds_dict: dict[str, tuple[float, float, float]],
        beta: float = 0.0,
        method: str = "trf",
        **kwargs: Any,
    ) -> MaterialParameterIdentifier:
        """
        Convenience factory creating an identifier from a dictionary:
            bounds_dict = {'youngs_modulus': (initial_value, lower_bound, upper_bound)}
        """
        return cls(bounds_dict, beta=beta, method=method, **kwargs)

    @staticmethod
    def _resolve_parameter_bounds(
        params: Sequence[ParameterBounds] | Sequence[str] | dict[str, tuple[float, float, float]],
    ) -> tuple[ParameterBounds, ...]:
        """Converts heterogeneous parameter inputs into tuple of validated ParameterBounds."""
        resolved: list[ParameterBounds] = []

        if isinstance(params, dict):
            for name, (init_val, lb, ub) in params.items():
                resolved.append(
                    ParameterBounds(
                        name=name,
                        initial_value=float(init_val),
                        lower_bound=float(lb),
                        upper_bound=float(ub),
                    )
                )
        elif isinstance(params, Sequence):
            for item in params:
                if isinstance(item, ParameterBounds):
                    resolved.append(item)
                elif isinstance(item, str):
                    # Default unbound parameter with initial=1.0
                    resolved.append(
                        ParameterBounds(
                            name=item,
                            initial_value=1.0,
                            lower_bound=1e-8,
                            upper_bound=np.inf,
                        )
                    )
                else:
                    raise TypeError(f"Unsupported parameter specification: {type(item)}")
        else:
            raise TypeError(f"parameters must be Sequence or dict, got {type(params)}")

        if len(resolved) == 0:
            raise ValueError("Parameter list cannot be empty.")

        return tuple(resolved)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Tuple of parameter names being identified."""
        return self._param_names

    @property
    def n_parameters(self) -> int:
        """Number of parameters P."""
        return len(self._param_bounds)

    @property
    def initial_parameters(self) -> dict[str, float]:
        """Initial parameter guess mapping."""
        return {p.name: p.initial_value for p in self._param_bounds}

    def identify(
        self,
        experimental_forces: np.ndarray | Sequence[float],
        forward_model: Callable[[dict[str, float] | np.ndarray], np.ndarray],
    ) -> IdentificationResult:
        """
        Executes parameter calibration against target experimental forces.

        Parameters:
        -----------
        experimental_forces : Sequence or np.ndarray of shape (M,)
            Target force responses observed at displacement / indentation steps.
        forward_model : Callable
            Forward simulation function taking either a dict {name: value} or
            a 1D parameter array theta in R^P and returning simulated forces in R^M.

        Returns:
        --------
        result : IdentificationResult
        """
        f_exp = np.asarray(experimental_forces, dtype=np.float64).ravel()
        if f_exp.size == 0:
            raise ValueError("experimental_forces cannot be empty.")
        M = f_exp.size

        # Resolve observation weights
        if self.weights is not None:
            w_arr = np.asarray(self.weights, dtype=np.float64).ravel()
            if w_arr.size != M:
                raise ValueError(
                    f"weights length ({w_arr.size}) must match experimental_forces length ({M})."
                )
            if np.any(w_arr <= 0.0):
                raise ValueError("All weights must be strictly positive.")
            sqrt_w = np.sqrt(w_arr)
        else:
            sqrt_w = np.ones(M, dtype=np.float64)

        x0 = self._x0.copy()
        bounds = (self._lower_bounds, self._upper_bounds)
        param_names = self._param_names
        beta = self.beta

        history: list[dict[str, float]] = []
        n_evals = 0

        # Parameter conversion helper
        def _to_params(x_vec: np.ndarray) -> dict[str, float]:
            return {name: float(val) for name, val in zip(param_names, x_vec)}

        # Objective residual function
        def _residual_fn(x_vec: np.ndarray) -> np.ndarray:
            nonlocal n_evals
            n_evals += 1

            p_dict = _to_params(x_vec)
            history.append(p_dict)

            try:
                # Support forward_model accepting dict or vector
                try:
                    f_sim = forward_model(p_dict)
                except TypeError:
                    f_sim = forward_model(x_vec)

                f_sim_arr = np.asarray(f_sim, dtype=np.float64).ravel()
                if f_sim_arr.size != M:
                    raise ValueError(
                        f"forward_model output size ({f_sim_arr.size}) must match "
                        f"experimental_forces ({M})."
                    )

                data_residual = sqrt_w * (f_sim_arr - f_exp)

                if beta > 0.0:
                    prior_residual = np.sqrt(beta) * (x_vec - x0)
                    return np.concatenate([data_residual, prior_residual])

                return data_residual

            except Exception as e:
                # Assign severe penalty on simulation breakdown (e.g. divergence)
                large_penalty = 1e8 * np.ones(M)
                if beta > 0.0:
                    large_penalty = np.concatenate([large_penalty, np.ones(len(x_vec))])
                return large_penalty

        t0 = time.perf_counter()

        if self.method == "L-BFGS-B":
            # Scalar objective minimization
            def _scalar_cost(x_vec: np.ndarray) -> float:
                r = _residual_fn(x_vec)
                return float(0.5 * np.sum(r**2))

            bnds = list(zip(self._lower_bounds, self._upper_bounds))
            opt_res = spopt.minimize(
                _scalar_cost,
                x0=x0,
                method="L-BFGS-B",
                bounds=bnds,
                tol=self.ftol,
                options={"maxfun": self.max_nfev} if self.max_nfev else {},
            )
            x_opt = opt_res.x
            success = opt_res.success
            message = opt_res.message
            n_iters = opt_res.nit
        else:
            # Nonlinear least squares (TRF, Dogbox, LM)
            bnds = bounds if self.method != "lm" else (-np.inf, np.inf)
            opt_res = spopt.least_squares(
                _residual_fn,
                x0=x0,
                bounds=bnds,
                method=self.method,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                max_nfev=self.max_nfev,
            )
            x_opt = opt_res.x
            success = opt_res.success
            message = opt_res.message
            n_iters = opt_res.nfev

        t_wall = time.perf_counter() - t0

        # Final evaluation at x_opt
        opt_params = _to_params(x_opt)
        try:
            f_sim_final = forward_model(opt_params)
        except TypeError:
            f_sim_final = forward_model(x_opt)
        f_sim_final = np.asarray(f_sim_final, dtype=np.float64).ravel()

        final_residuals = f_sim_final - f_exp
        rms_err = float(np.sqrt(np.mean(final_residuals**2)))
        norm_exp = np.linalg.norm(f_exp)
        rel_err = float(np.linalg.norm(final_residuals) / norm_exp) if norm_exp > 0 else rms_err
        cost = float(0.5 * np.sum(final_residuals**2))

        return IdentificationResult(
            parameters=opt_params,
            initial_parameters=self.initial_parameters,
            simulated_forces=f_sim_final,
            experimental_forces=f_exp,
            residuals=final_residuals,
            rms_error=rms_err,
            relative_error=rel_err,
            iterations=n_iters,
            n_evaluations=n_evals,
            cost=cost,
            success=bool(success),
            message=str(message),
            wall_time_seconds=t_wall,
            parameter_history=history,
        )

    @classmethod
    def calibrate_linear_elasticity(
        cls,
        experimental_forces: np.ndarray | Sequence[float],
        baseline_forces: np.ndarray | Sequence[float],
        baseline_modulus: float,
        beta: float = 0.0,
    ) -> IdentificationResult:
        """
        Direct exact calibration of Young's modulus E for linear elastic structures.

        In linear elasticity, structural stiffness K(E) and reaction forces F(E)
        scale strictly linearly with Young's modulus:
            F_sim(E) = (E / E_0) * F_0

        Minimizing 1/2 * sum ( (E/E_0)*F_0 - F_exp )^2 + beta/2 * (E - E_0)^2
        has the exact closed-form solution:
            E* = ( sum(F_0 * F_exp) / E_0 + beta * E_0 ) / ( sum(F_0^2) / E_0^2 + beta )
        which evaluates in < 1 microsecond to machine precision!

        Parameters:
        -----------
        experimental_forces : Sequence or np.ndarray of shape (M,)
            Measured experimental forces at test steps.
        baseline_forces : Sequence or np.ndarray of shape (M,)
            Simulated forces computed with baseline modulus E_0.
        baseline_modulus : float
            Baseline Young's modulus E_0.
        beta : float, default=0.0
            Prior regularization weight.

        Returns:
        --------
        result : IdentificationResult
        """
        t0 = time.perf_counter()
        f_exp = np.asarray(experimental_forces, dtype=np.float64).ravel()
        f_0 = np.asarray(baseline_forces, dtype=np.float64).ravel()
        E0 = float(baseline_modulus)

        if f_exp.size != f_0.size:
            raise ValueError(
                f"experimental_forces size ({f_exp.size}) must match baseline_forces ({f_0.size})."
            )

        # Closed-form linear least squares solution
        s_0 = f_0 / E0  # Force per unit Young's modulus: F = E * s_0
        numerator = float(np.sum(s_0 * f_exp) + beta * E0)
        denominator = float(np.sum(s_0**2) + beta)

        if denominator <= 0.0:
            raise RuntimeError("Degenerate linear calibration: baseline forces are identically zero.")

        E_opt = numerator / denominator
        t_wall = time.perf_counter() - t0

        f_sim_opt = E_opt * s_0
        residuals = f_sim_opt - f_exp
        rms_err = float(np.sqrt(np.mean(residuals**2)))
        norm_exp = np.linalg.norm(f_exp)
        rel_err = float(np.linalg.norm(residuals) / norm_exp) if norm_exp > 0 else rms_err

        return IdentificationResult(
            parameters={"youngs_modulus": E_opt},
            initial_parameters={"youngs_modulus": E0},
            simulated_forces=f_sim_opt,
            experimental_forces=f_exp,
            residuals=residuals,
            rms_error=rms_err,
            relative_error=rel_err,
            iterations=1,
            n_evaluations=1,
            cost=float(0.5 * np.sum(residuals**2)),
            success=True,
            message="Closed-form linear least-squares solution converged.",
            wall_time_seconds=t_wall,
        )
