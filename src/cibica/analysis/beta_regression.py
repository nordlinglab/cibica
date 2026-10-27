# Copyright 2026 Esteban Román Catafau and Torbjörn E. M. Nordling
# SPDX-License-Identifier: Apache-2.0
"""Paired beta regression for a bounded outcome on two matched methods.

The paired scores are stacked into one long table with *method* as a fixed
effect and *frame* as a blocking factor, mapped into ``(0, 1)`` with the
Smithson-Verkuilen transform and modelled by maximum likelihood with a logit
link on the mean and a constant precision ``phi`` on a log link (Ferrari &
Cribari-Neto, 2004), the specification Appendix A of the manuscript gives.

The fit is self-contained: the log-likelihood, its score and its observed
Hessian are coded in closed form, so the estimator needs only SciPy. Wald
standard errors come from the inverse of the negative observed Hessian at the
maximum, which is what ``statsmodels``' ``BetaModel`` approximates
numerically; the two agree to the optimiser tolerance.

Two specifications are available because they trade different risks:

``frame``
    The blocked specification Appendix A states, one dummy per frame. With
    two observations per frame the frame effects are incidental parameters,
    so ``phi`` is not consistently estimated even though the method
    coefficient stays the quantity of interest.
``pooled``
    Method only. Free of the incidental-parameter problem but blind to the
    pairing, so its standard error is the more conservative of the two.

References:
    Ferrari, S., & Cribari-Neto, F. (2004). Beta regression for modelling rates
        and proportions. *Journal of Applied Statistics*, 31(7), 799-815.
        https://doi.org/10.1080/0266476042000214501
    Smithson, M., & Verkuilen, J. (2006). A better lemon squeezer?
        Maximum-likelihood regression with beta-distributed dependent
        variables. *Psychological Methods*, 11(1), 54-71.
        https://doi.org/10.1037/1082-989X.11.1.54
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, logit, ndtr, polygamma

#: Specifications accepted by :func:`beta_regression`.
SPECIFICATIONS: tuple[str, ...] = ("frame", "pooled")

#: Normal quantile of the two-sided 95% Wald interval.
_Z_95 = 1.959963984540054

#: Fitted means are kept this far from the bounds so the likelihood stays finite.
_MU_EPS = 1e-10

#: Newton decrement below which the fit counts as converged; it bounds the
#: log-likelihood still to be gained, so ``1e-6`` is far inside rounding.
_DECREMENT_TOL = 1e-6


def smithson_verkuilen_transform(values, n: int | None = None) -> np.ndarray:
    """Map ``[0, 1]`` into the open interval ``(0, 1)``.

    Args:
        values: Scores in ``[0, 1]``.
        n: Sample size used for the adjustment; defaults to ``len(values)``.

    Returns:
        ``(values * (n - 1) + 0.5) / n``, strictly inside ``(0, 1)``.
    """
    values = np.asarray(values, float)
    size = values.size if n is None else n
    return np.asarray((values * (size - 1) + 0.5) / size, float)


def smithson_verkuilen_inverse(values, n: int) -> np.ndarray:
    """Invert :func:`smithson_verkuilen_transform` back to the original scale."""
    values = np.asarray(values, float)
    return np.asarray((values * n - 0.5) / (n - 1), float)


def boundary_point_mass(values) -> dict:
    """Count and fraction of observations sitting exactly on ``0`` or ``1``.

    A beta likelihood assigns zero density to the bounds, so substantial mass
    there calls for a zero/one-inflated model instead of plain beta regression.

    Args:
        values: Scores in ``[0, 1]``.

    Returns:
        dict: ``n``, ``n_at_zero``, ``n_at_one``, ``fraction_at_zero`` and
        ``fraction_at_one``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    values = np.asarray(values, float)
    if values.size == 0:
        raise ValueError("values must contain at least one observation")
    n_zero = int(np.sum(values <= 0.0))
    n_one = int(np.sum(values >= 1.0))
    n = int(values.size)
    return {
        "n": n,
        "n_at_zero": n_zero,
        "n_at_one": n_one,
        "fraction_at_zero": n_zero / n,
        "fraction_at_one": n_one / n,
    }


def _design(n_frames: int, spec: str) -> np.ndarray:
    """Design matrix for the stacked paired data.

    Row order is the baseline block followed by the focal block, so column 1 is
    the method indicator (0 = baseline, 1 = focal).

    Raises:
        ValueError: If ``spec`` is not in :data:`SPECIFICATIONS`.
    """
    if spec not in SPECIFICATIONS:
        raise ValueError(f"spec must be one of {SPECIFICATIONS}, got {spec!r}")
    n_obs = 2 * n_frames
    const = np.ones(n_obs)
    method = np.concatenate([np.zeros(n_frames), np.ones(n_frames)])
    if spec == "pooled":
        return np.column_stack([const, method])
    eye = np.eye(n_frames)[:, 1:]  # first frame absorbed by the intercept
    return np.column_stack([const, method, np.vstack([eye, eye])])


class _BetaLikelihood:
    """Log-likelihood of a beta regression with logit mean and log precision.

    Parameters are ``(beta, theta)`` with ``phi = exp(theta)``; every method
    returns the *negative* quantity so SciPy's minimisers can be used directly.
    """

    def __init__(self, y: np.ndarray, exog: np.ndarray) -> None:
        self.y = y
        self.exog = exog
        self.y_star = logit(y)
        self.log_y = np.log(y)
        self.log_1my = np.log1p(-y)

    def _state(self, params: np.ndarray):
        beta, theta = params[:-1], params[-1]
        phi = float(np.exp(theta))
        mu = np.clip(expit(self.exog @ beta), _MU_EPS, 1.0 - _MU_EPS)
        return phi, mu, mu * phi, (1.0 - mu) * phi

    def value(self, params: np.ndarray) -> float:
        phi, mu, a, b = self._state(params)
        ll = (
            gammaln(phi)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * self.log_y
            + (b - 1.0) * self.log_1my
        )
        return -float(np.sum(ll))

    def gradient(self, params: np.ndarray) -> np.ndarray:
        phi, mu, a, b = self._state(params)
        residual = self.y_star - (digamma(a) - digamma(b))
        weight = mu * (1.0 - mu)
        score_beta = self.exog.T @ (phi * residual * weight)
        score_phi = float(
            np.sum(mu * residual + self.log_1my - digamma(b) + digamma(phi))
        )
        return -np.append(score_beta, phi * score_phi)

    def hessian(self, params: np.ndarray) -> np.ndarray:
        phi, mu, a, b = self._state(params)
        psi1_a, psi1_b = polygamma(1, a), polygamma(1, b)
        residual = self.y_star - (digamma(a) - digamma(b))
        weight = mu * (1.0 - mu)
        curvature = weight * (1.0 - 2.0 * mu)
        d_bb = phi * (-phi * (psi1_a + psi1_b) * weight**2 + residual * curvature)
        h_bb = self.exog.T @ (d_bb[:, None] * self.exog)
        w = mu * psi1_a - (1.0 - mu) * psi1_b
        h_bphi = self.exog.T @ (weight * (residual - phi * w))
        h_phiphi = float(
            np.sum(-(mu**2) * psi1_a - (1.0 - mu) ** 2 * psi1_b + polygamma(1, phi))
        )
        score_phi = float(
            np.sum(mu * residual + self.log_1my - digamma(b) + digamma(phi))
        )
        k = params.size
        hess = np.empty((k, k))
        hess[:-1, :-1] = h_bb
        hess[:-1, -1] = hess[-1, :-1] = phi * h_bphi
        hess[-1, -1] = phi**2 * h_phiphi + phi * score_phi
        return -hess


def _starting_values(y: np.ndarray, exog: np.ndarray) -> np.ndarray:
    """Ferrari & Cribari-Neto's (2004) starting values: OLS on the logit scale."""
    beta, *_ = np.linalg.lstsq(exog, logit(y), rcond=None)
    eta = exog @ beta
    mu = np.clip(expit(eta), 1e-6, 1.0 - 1e-6)
    n, k = exog.shape
    residual_var = float(np.sum((logit(y) - eta) ** 2)) / max(n - k, 1)
    sigma2 = residual_var * (mu * (1.0 - mu)) ** 2
    phi0 = max(float(np.mean(mu * (1.0 - mu) / sigma2 - 1.0)), 1.0)
    return np.append(beta, np.log(phi0))


def beta_regression(
    focal, baseline, spec: str = "frame", maxiter: int = 5_000, tol: float = 1e-6
) -> dict:
    """Fit a paired beta regression of a bounded score on method.

    Args:
        focal: Per-frame scores for the focal method (e.g. CIBICA).
        baseline: Per-frame scores for the baseline, in the same frame order.
        spec: ``"frame"`` (frame blocking factor) or ``"pooled"``.
        maxiter: Maximum optimiser iterations.
        tol: Gradient tolerance of the trust-region Newton optimiser; the
            returned ``converged`` flag uses the Newton decrement instead, so
            a fit that stalls on this tolerance at the maximum still counts.

    Returns:
        dict: the method ``coefficient`` on the logit scale, its Wald
        ``std_error``, ``z_value``, two-sided ``p_value`` and 95% interval
        (``ci_lo``, ``ci_hi``); the precision ``phi``; the back-transformed
        marginal means (``marginal_focal``, ``marginal_baseline``) and their
        difference ``marginal_improvement`` on the original scale;
        ``log_likelihood``, ``newton_decrement``, ``n_obs``, ``n_frames``,
        ``spec`` and ``converged``.

    Raises:
        ValueError: If the two arrays differ in length or ``spec`` is unknown.
    """
    focal = np.asarray(focal, float)
    baseline = np.asarray(baseline, float)
    if focal.shape != baseline.shape:
        raise ValueError("focal and baseline must cover the same frames")
    n_frames = int(focal.size)
    n_obs = 2 * n_frames
    exog = _design(n_frames, spec)
    y = smithson_verkuilen_transform(np.concatenate([baseline, focal]), n_obs)

    model = _BetaLikelihood(y, exog)
    result = minimize(
        model.value,
        _starting_values(y, exog),
        jac=model.gradient,
        hess=model.hessian,
        method="trust-exact",
        options={"maxiter": maxiter, "gtol": tol},
    )
    params = result.x
    gradient = model.gradient(params)
    information = model.hessian(params)  # negative Hessian of the log-likelihood
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:  # pragma: no cover - singular at the optimum
        covariance = np.linalg.pinv(information)
    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    # Scale-free convergence check: the Newton decrement ``g' H^-1 g`` bounds
    # the log-likelihood gain still available, whereas the raw gradient norm
    # grows with ``phi`` and can sit above any fixed tolerance at the maximum.
    decrement = float(gradient @ covariance @ gradient)
    converged = bool(result.success) or (0.0 <= decrement < _DECREMENT_TOL)

    coefficient = float(params[1])
    std_error = float(std_errors[1])
    z_value = coefficient / std_error if std_error > 0 else float("inf")
    fitted = smithson_verkuilen_inverse(expit(exog @ params[:-1]), n_obs)
    marginal_baseline = float(fitted[:n_frames].mean())
    marginal_focal = float(fitted[n_frames:].mean())
    return {
        "spec": spec,
        "n_obs": n_obs,
        "n_frames": n_frames,
        "coefficient": coefficient,
        "std_error": std_error,
        "z_value": float(z_value),
        "p_value": float(2.0 * ndtr(-abs(z_value))),
        "ci_lo": coefficient - _Z_95 * std_error,
        "ci_hi": coefficient + _Z_95 * std_error,
        "phi": float(np.exp(params[-1])),
        "marginal_focal": marginal_focal,
        "marginal_baseline": marginal_baseline,
        "marginal_improvement": marginal_focal - marginal_baseline,
        "log_likelihood": -float(result.fun),
        "newton_decrement": decrement,
        "converged": converged,
    }


__all__ = [
    "SPECIFICATIONS",
    "beta_regression",
    "boundary_point_mass",
    "smithson_verkuilen_inverse",
    "smithson_verkuilen_transform",
]
