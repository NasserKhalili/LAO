"""Environmental forcing models used by LAO.

Continuous turbulence, mean vertical profiles, discrete gusts, and
long-term wind-speed distributions are represented separately because they
play different modelling roles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

CONTINUOUS_TURBULENCE = "continuous_turbulence"
MEAN_PROFILE = "mean_profile"
DISCRETE_GUST = "discrete_gust"
LONG_TERM_DISTRIBUTION = "long_term_distribution"


class ForcingProcess(ABC):
    """A scalar environmental process advanced once per algorithmic step."""

    name: str = "base"
    role: str = CONTINUOUS_TURBULENCE
    reference: str = ""

    @abstractmethod
    def step(self, tau: float, rng: np.random.Generator) -> float:
        """Advance one step and return the normalised field state in [0, 1]."""

    def reset(self) -> None:
        return None


class DrydenForcing(ForcingProcess):
    """Dryden continuous turbulence as a first-order filter on white noise.

    The Dryden longitudinal spectrum is rational,

        Phi_u(Omega) = (2 sigma_u^2 L_u / pi) / (1 + (L_u Omega)^2),

    so it is realised exactly by the first-order transfer function
    ``H_u(s) = sigma_u sqrt(2 L_u / (pi V)) / (1 + L_u s / V)``.  The filter is
    integrated with an explicit Euler step at the algorithmic step rate; the
    stationary standard deviation of the discretised state is reported by
    :meth:`stationary_std` so the realised intensity can be checked.
    """

    name = "Dryden"
    role = CONTINUOUS_TURBULENCE
    reference = "MIL-HDBK-1797 (1997), Sec. 5.4.2"

    def __init__(self, sigma_u: float = 0.30, length_scale: float = 0.30,
                 mean_speed: float = 0.35, dt: float = 0.05) -> None:
        self.sigma_u = float(sigma_u)
        self.length_scale = float(length_scale)
        self.mean_speed = float(mean_speed)
        self.dt = float(dt)
        self._state = 0.0

    def reset(self) -> None:
        self._state = 0.0

    @property
    def _tau_filter(self) -> float:
        return self.length_scale / max(self.mean_speed, 1e-12)

    def stationary_std(self) -> float:
        gain = self.sigma_u * np.sqrt(2.0 * self.length_scale / (np.pi * max(self.mean_speed, 1e-12)))
        a = self.dt / self._tau_filter
        if a >= 2.0:  # Euler step is unstable; report the analytic bound
            return float("inf")
        return float(gain * a / np.sqrt(a * (2.0 - a)))

    def step(self, tau: float, rng: np.random.Generator) -> float:
        gain = self.sigma_u * np.sqrt(2.0 * self.length_scale / (np.pi * max(self.mean_speed, 1e-12)))
        filter_tau = self._tau_filter
        noise = rng.normal(0.0, 1.0)
        self._state += (-self._state / filter_tau + gain * noise / filter_tau) * self.dt
        return float(np.clip(self.mean_speed + self._state, 0.0, 1.0))


class VonKarmanForcing(ForcingProcess):
    """Von Karman turbulence via Shinozuka spectral superposition.

    The Von Karman spectrum has a non-rational 5/6 exponent and therefore no
    exact finite-order filter, so it is realised as a sum of harmonics whose
    amplitudes follow the analytic PSD and whose phases are drawn once.
    """

    name = "VonKarman"
    role = CONTINUOUS_TURBULENCE
    reference = "MIL-HDBK-1797 (1997), Sec. 5.4.1; Von Karman (1948)"

    def __init__(self, sigma_u: float = 0.30, length_scale: float = 0.40,
                 mean_speed: float = 0.35, n_harmonics: int = 32,
                 omega_max: float = 40.0) -> None:
        self.sigma_u = float(sigma_u)
        self.length_scale = float(length_scale)
        self.mean_speed = float(mean_speed)
        self.n_harmonics = int(n_harmonics)
        self.omega_max = float(omega_max)
        self._omega = np.linspace(self.omega_max / self.n_harmonics, self.omega_max, self.n_harmonics)
        self._phase: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._phase = None

    def _psd(self, omega: np.ndarray) -> np.ndarray:
        return (2.0 * self.sigma_u ** 2 * self.length_scale
                / (1.0 + (1.339 * self.length_scale * omega) ** 2) ** (5.0 / 6.0))

    def step(self, tau: float, rng: np.random.Generator) -> float:
        if self._phase is None:
            self._phase = rng.uniform(0.0, 2.0 * np.pi, self.n_harmonics)
        d_omega = self._omega[1] - self._omega[0]
        amplitudes = np.sqrt(2.0 * self._psd(self._omega) * d_omega)
        fluctuation = float(np.sum(amplitudes * np.cos(self._omega * tau + self._phase)))
        return float(np.clip(self.mean_speed + fluctuation, 0.0, 1.0))


class OneCosineGust:
    """1-cosine discrete gust with stochastic arrival (FAA 14 CFR 25.341).

        U(s) = (U_max / 2) (1 - cos(pi s / T_g)),   0 <= s <= T_g

    Arrival is a Bernoulli trial per step with probability ``arrival_rate``;
    while a gust is active the waveform advances in normalised time.
    """

    name = "OneCosine"
    role = DISCRETE_GUST
    reference = "FAA 14 CFR 25.341; ESDU 83004"

    def __init__(self, peak: float = 1.0, duration: float = 0.02,
                 arrival_rate: float = 0.05, step_size: float = 0.004) -> None:
        self.peak = float(peak)
        self.duration = float(duration)
        self.arrival_rate = float(arrival_rate)
        self.step_size = float(step_size)
        self._active = False
        self._elapsed = 0.0

    def reset(self) -> None:
        self._active = False
        self._elapsed = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def step(self, tau: float, rng: np.random.Generator) -> float:
        if not self._active and rng.random() < self.arrival_rate:
            self._active = True
            self._elapsed = 0.0
        if not self._active:
            return 0.0
        amplitude = (self.peak / 2.0) * (1.0 - np.cos(np.pi * self._elapsed / self.duration))
        self._elapsed += self.step_size
        if self._elapsed >= self.duration:
            self._active = False
            self._elapsed = 0.0
        return float(np.clip(amplitude, 0.0, self.peak))


class PowerLawProfile:
    """Power-law vertical profile, ``U(z) = U_ref (z / z_ref)^alpha`` (ASCE 7-16)."""

    name = "PowerLaw"
    role = MEAN_PROFILE
    reference = "ASCE/SEI 7-16 (2017), Sec. 26.9"

    def __init__(self, alpha: float = 1.0 / 7.0) -> None:
        self.alpha = float(alpha)

    def __call__(self, z: np.ndarray | float) -> np.ndarray | float:
        return np.clip(np.asarray(z, dtype=float), 1e-3, 1.0) ** self.alpha


class LogarithmicProfile:
    """Logarithmic vertical profile, normalised to 1 at the canopy top."""

    name = "Logarithmic"
    role = MEAN_PROFILE
    reference = "Stull (1988)"

    def __init__(self, roughness: float = 0.01) -> None:
        self.roughness = float(roughness)

    def __call__(self, z: np.ndarray | float) -> np.ndarray | float:
        zz = np.clip(np.asarray(z, dtype=float), 2.0 * self.roughness, 1.0)
        top = np.log(1.0 / self.roughness)
        return np.log(zz / self.roughness) / top


class WeibullMarginal:
    """Weibull long-term wind-speed marginal (IEC 61400-1).

    Provided only for the environmental-model comparison.  It is a marginal
    distribution over long observation windows, so using it as a per-step
    process discards all temporal correlation; that is exactly why it is
    labelled ``LONG_TERM_DISTRIBUTION`` and not offered as default forcing.
    """

    name = "Weibull"
    role = LONG_TERM_DISTRIBUTION
    reference = "IEC 61400-1 (2005)"

    def __init__(self, shape: float = 2.0, scale: float = 0.45) -> None:
        self.shape = float(shape)
        self.scale = float(scale)

    def reset(self) -> None:
        return None

    def step(self, tau: float, rng: np.random.Generator) -> float:
        return float(np.clip(rng.weibull(self.shape) * self.scale, 0.0, 1.0))


FORCING_PROCESSES: Dict[str, type] = {
    "Dryden": DrydenForcing,
    "VonKarman": VonKarmanForcing,
    "Weibull": WeibullMarginal,
}
PROFILES: Dict[str, type] = {
    "PowerLaw": PowerLawProfile,
    "Logarithmic": LogarithmicProfile,
}
PROCESS_ROLES: Dict[str, str] = {
    "Dryden": CONTINUOUS_TURBULENCE,
    "VonKarman": CONTINUOUS_TURBULENCE,
    "Weibull": LONG_TERM_DISTRIBUTION,
    "OneCosine": DISCRETE_GUST,
    "PowerLaw": MEAN_PROFILE,
    "Logarithmic": MEAN_PROFILE,
}


@dataclass
class EnvironmentState:
    """The environmental field at one algorithmic step."""

    tau: float
    continuous: float
    gust: float
    gust_active: bool


class EnvironmentField:
    """Common environmental state shared by the whole population.

    One :meth:`step` per algorithmic iteration produces the field state; the
    per-candidate exposure is then a deterministic function of that state,
    the candidate's canopy height ``z_i`` and its own susceptibility draw.
    """

    def __init__(
        self,
        forcing: str = "Dryden",
        profile: str = "PowerLaw",
        gust_peak: float = 1.0,
        gust_duration: float = 0.02,
        gust_arrival_rate: float = 0.05,
        gust_step: float = 0.004,
        exposure_noise: float = 0.05,
        enable_gust: bool = True,
        forcing_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        if forcing not in FORCING_PROCESSES:
            raise ValueError(f"unknown forcing process: {forcing}")
        if profile not in PROFILES:
            raise ValueError(f"unknown vertical profile: {profile}")
        self.forcing_name = forcing
        self.profile_name = profile
        self.process = FORCING_PROCESSES[forcing](**(forcing_kwargs or {}))
        self.profile = PROFILES[profile]()
        self.enable_gust = bool(enable_gust)
        self.gust = OneCosineGust(gust_peak, gust_duration, gust_arrival_rate, gust_step)
        self.exposure_noise = float(exposure_noise)
        self.state = EnvironmentState(0.0, 0.0, 0.0, False)

    def reset(self) -> None:
        self.process.reset()
        self.gust.reset()
        self.state = EnvironmentState(0.0, 0.0, 0.0, False)

    def step(self, tau: float, rng: np.random.Generator) -> EnvironmentState:
        continuous = self.process.step(tau, rng)
        gust = self.gust.step(tau, rng) if self.enable_gust else 0.0
        self.state = EnvironmentState(float(tau), float(continuous), float(gust), self.gust.active)
        return self.state

    def exposure(self, heights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Per-candidate exposure ``U_i = clip(W_t h(z_i) + noise, 0, 1)``."""
        weight = np.asarray(self.profile(heights), dtype=float)
        jitter = rng.normal(0.0, self.exposure_noise, size=weight.shape) if self.exposure_noise > 0 else 0.0
        return np.clip(self.state.continuous * weight + jitter, 0.0, 1.0)

    def describe(self) -> Dict[str, object]:
        return {
            "forcing": self.forcing_name,
            "forcing_role": PROCESS_ROLES[self.forcing_name],
            "forcing_reference": getattr(self.process, "reference", ""),
            "profile": self.profile_name,
            "profile_role": PROCESS_ROLES[self.profile_name],
            "profile_reference": getattr(self.profile, "reference", ""),
            "gust_enabled": self.enable_gust,
            "gust_role": DISCRETE_GUST,
            "gust_reference": self.gust.reference,
            "exposure_noise": self.exposure_noise,
        }
