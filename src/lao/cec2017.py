"""CEC 2017 single-objective bound-constrained benchmark interface.

The official support data are expected under ``cec2017_data/`` and are not
redistributed with this repository. The reference implementation remains the
authoritative computational definition used by the benchmark interface.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by both code paths
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


SUITE_NAME = "CEC2017"
IS_OFFICIAL_CEC2017 = True
DATA_DIR = Path(__file__).resolve().parent / "cec2017_data"
#: Parsed support data is cached as .npy beside the official text files.
#: The text files remain authoritative; the cache is rebuilt from them on
#: demand and is safe to delete.
_CACHE_DIR = DATA_DIR / "_parsed"
SUPPORTED_DIMENSIONS = (2, 10, 20, 30, 50, 100)
#: Dimensions for which every hybrid/composition support file exists.
FULL_DIMENSIONS = (10, 30, 50, 100)
#: F2 is excluded by the competition itself for numerical instability.
EXCLUDED_FUNCTIONS = (2,)
EXPERIMENTAL_FUNCTIONS = tuple(n for n in range(1, 31) if n not in EXCLUDED_FUNCTIONS)

INF = 1.0e99
_SCHWEFEL_OFFSET = 4.209687462275036e2
_SCHWEFEL_CONST = 4.189828872724338e2


# --------------------------------------------------------------------- #
# Primitives (ports of shiftfunc / rotatefunc / sr_func)
# --------------------------------------------------------------------- #
@njit(cache=True)
def _rotate(src, dst, nx, Mr):
    for i in range(nx):
        acc = 0.0
        base = i * nx
        for j in range(nx):
            acc += src[j] * Mr[base + j]
        dst[i] = acc


@njit(cache=True)
def _sr_func(x, sr_x, nx, Os, Mr, sh_rate, s_flag, r_flag, y):
    if s_flag == 1:
        if r_flag == 1:
            for i in range(nx):
                y[i] = (x[i] - Os[i]) * sh_rate
            _rotate(y, sr_x, nx, Mr)
        else:
            for i in range(nx):
                sr_x[i] = (x[i] - Os[i]) * sh_rate
    else:
        if r_flag == 1:
            for i in range(nx):
                y[i] = x[i] * sh_rate
            _rotate(y, sr_x, nx, Mr)
        else:
            for i in range(nx):
                sr_x[i] = x[i] * sh_rate


# --------------------------------------------------------------------- #
# Basic functions
# --------------------------------------------------------------------- #
@njit(cache=True)
def _ellips(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx):
        total += 10.0 ** (6.0 * i / (nx - 1)) * z[i] * z[i]
    return total


@njit(cache=True)
def _sum_diff_pow(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx):
        total += abs(z[i]) ** (i + 1)
    return total


@njit(cache=True)
def _zakharov(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(nx):
        sum1 += z[i] * z[i]
        sum2 += 0.5 * (i + 1) * z[i]
    return sum1 + sum2 ** 2 + sum2 ** 4


@njit(cache=True)
def _levy(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    w_last = 1.0 + (z[nx - 1] - 1.0) / 4.0
    w_first = 1.0 + (z[0] - 1.0) / 4.0
    term1 = np.sin(np.pi * w_first) ** 2
    term3 = (w_last - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w_last) ** 2)
    total = 0.0
    for i in range(nx - 1):
        wi = 1.0 + (z[i] - 1.0) / 4.0
        total += (wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wi + 1.0) ** 2)
    return term1 + total + term3


@njit(cache=True)
def _bent_cigar(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = z[0] * z[0]
    for i in range(1, nx):
        total += 1.0e6 * z[i] * z[i]
    return total


@njit(cache=True)
def _discus(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = 1.0e6 * z[0] * z[0]
    for i in range(1, nx):
        total += z[i] * z[i]
    return total


@njit(cache=True)
def _rosenbrock(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag, y)
    z[0] += 1.0
    total = 0.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        total += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    return total


@njit(cache=True)
def _schaffer_f7(x, nx, Os, Mr, s_flag, r_flag, y, z):
    # Faithful to the reference: the sum reads the pre-rotation buffer y.
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx - 1):
        zi = (y[i] * y[i] + y[i + 1] * y[i + 1]) ** 0.5
        z[i] = zi
        tmp = np.sin(50.0 * zi ** 0.2)
        total += zi ** 0.5 + zi ** 0.5 * tmp * tmp
    return total * total / (nx - 1) / (nx - 1)


@njit(cache=True)
def _ackley(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(nx):
        sum1 += z[i] * z[i]
        sum2 += np.cos(2.0 * np.pi * z[i])
    sum1 = -0.2 * np.sqrt(sum1 / nx)
    sum2 /= nx
    return np.e - 20.0 * np.exp(sum1) - np.exp(sum2) + 20.0


@njit(cache=True)
def _weierstrass(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 0.5 / 100.0, s_flag, r_flag, y)
    a = 0.5
    b = 3.0
    k_max = 20
    total = 0.0
    sum2 = 0.0
    for i in range(nx):
        acc = 0.0
        sum2 = 0.0
        for j in range(k_max + 1):
            aj = a ** j
            bj = b ** j
            acc += aj * np.cos(2.0 * np.pi * bj * (z[i] + 0.5))
            sum2 += aj * np.cos(2.0 * np.pi * bj * 0.5)
        total += acc
    return total - nx * sum2


@njit(cache=True)
def _griewank(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag, y)
    s = 0.0
    p = 1.0
    for i in range(nx):
        s += z[i] * z[i]
        p *= np.cos(z[i] / np.sqrt(1.0 + i))
    return 1.0 + s / 4000.0 - p


@njit(cache=True)
def _rastrigin(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx):
        total += z[i] * z[i] - 10.0 * np.cos(2.0 * np.pi * z[i]) + 10.0
    return total


@njit(cache=True)
def _step_rastrigin(x, nx, Os, Mr, s_flag, r_flag, y, z):
    # Reference behaviour: the rounding below is applied to the stale y
    # buffer and is then discarded by sr_func, which rewrites y in full.
    for i in range(nx):
        if abs(y[i] - Os[i]) > 0.5:
            y[i] = Os[i] + np.floor(2.0 * (y[i] - Os[i]) + 0.5) / 2.0
    _sr_func(x, z, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx):
        total += z[i] * z[i] - 10.0 * np.cos(2.0 * np.pi * z[i]) + 10.0
    return total


@njit(cache=True)
def _schwefel(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx):
        z[i] += _SCHWEFEL_OFFSET
        zi = z[i]
        if zi > 500.0:
            rem = np.fmod(zi, 500.0)
            total -= (500.0 - rem) * np.sin((500.0 - rem) ** 0.5)
            tmp = (zi - 500.0) / 100.0
            total += tmp * tmp / nx
        elif zi < -500.0:
            rem = np.fmod(abs(zi), 500.0)
            total -= (-500.0 + rem) * np.sin((500.0 - rem) ** 0.5)
            tmp = (zi + 500.0) / 100.0
            total += tmp * tmp / nx
        else:
            total -= zi * np.sin(abs(zi) ** 0.5)
    return total + _SCHWEFEL_CONST * nx


@njit(cache=True)
def _katsuura(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag, y)
    total = 1.0
    tmp3 = (1.0 * nx) ** 1.2
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * z[i]
            temp += abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
        total *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    tmp1 = 10.0 / nx / nx
    return total * tmp1 - tmp1


@njit(cache=True)
def _bi_rastrigin(x, nx, Os, Mr, s_flag, r_flag, y, z, tmpx):
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * (nx + 20.0) ** 0.5 - 8.2)
    mu1 = -(((mu0 * mu0 - d) / s) ** 0.5)
    if s_flag == 1:
        for i in range(nx):
            y[i] = x[i] - Os[i]
    else:
        for i in range(nx):
            y[i] = x[i]
    for i in range(nx):
        y[i] *= 10.0 / 100.0
    for i in range(nx):
        tmpx[i] = 2.0 * y[i]
        if Os[i] < 0.0:
            tmpx[i] *= -1.0
    for i in range(nx):
        z[i] = tmpx[i]
        tmpx[i] += mu0
    tmp1 = 0.0
    tmp2 = 0.0
    for i in range(nx):
        t = tmpx[i] - mu0
        tmp1 += t * t
        t = tmpx[i] - mu1
        tmp2 += t * t
    tmp2 *= s
    tmp2 += d * nx
    tmp = 0.0
    if r_flag == 1:
        _rotate(z, y, nx, Mr)
        for i in range(nx):
            tmp += np.cos(2.0 * np.pi * y[i])
    else:
        for i in range(nx):
            tmp += np.cos(2.0 * np.pi * z[i])
    base = tmp1 if tmp1 < tmp2 else tmp2
    return base + 10.0 * (nx - tmp)


@njit(cache=True)
def _grie_rosen(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag, y)
    z[0] += 1.0
    total = 0.0
    for i in range(nx - 1):
        z[i + 1] += 1.0
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        total += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    total += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    return total


@njit(cache=True)
def _escaffer6(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    total = 0.0
    for i in range(nx - 1):
        q = z[i] * z[i] + z[i + 1] * z[i + 1]
        t1 = np.sin(np.sqrt(q))
        t1 = t1 * t1
        t2 = 1.0 + 0.001 * q
        total += 0.5 + (t1 - 0.5) / (t2 * t2)
    q = z[nx - 1] * z[nx - 1] + z[0] * z[0]
    t1 = np.sin(np.sqrt(q))
    t1 = t1 * t1
    t2 = 1.0 + 0.001 * q
    total += 0.5 + (t1 - 0.5) / (t2 * t2)
    return total


@njit(cache=True)
def _happycat(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag, y)
    alpha = 1.0 / 8.0
    r2 = 0.0
    sum_z = 0.0
    for i in range(nx):
        z[i] = z[i] - 1.0
        r2 += z[i] * z[i]
        sum_z += z[i]
    return abs(r2 - nx) ** (2.0 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


@njit(cache=True)
def _hgbat(x, nx, Os, Mr, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag, y)
    alpha = 1.0 / 4.0
    r2 = 0.0
    sum_z = 0.0
    for i in range(nx):
        z[i] = z[i] - 1.0
        r2 += z[i] * z[i]
        sum_z += z[i]
    return abs(r2 * r2 - sum_z * sum_z) ** (2.0 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


# --------------------------------------------------------------------- #
# Hybrid helpers
# --------------------------------------------------------------------- #
@njit(cache=True)
def _hybrid_layout(gp, nx, g_nx, g_off):
    cf_num = gp.shape[0]
    tmp = 0
    for i in range(cf_num - 1):
        g_nx[i] = int(np.ceil(gp[i] * nx))
        tmp += g_nx[i]
    g_nx[cf_num - 1] = nx - tmp
    g_off[0] = 0
    for i in range(1, cf_num):
        g_off[i] = g_off[i - 1] + g_nx[i - 1]


@njit(cache=True)
def _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z):
    _sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag, y)
    # y[i] = z[S[i] - 1]; z is fully written first, so the gather is safe.
    for i in range(nx):
        y[i] = z[SS[i] - 1]


@njit(cache=True)
def _hf01(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.4, 0.4])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _zakharov(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _rosenbrock(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf02(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.3, 0.3, 0.4])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _ellips(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _schwefel(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _bent_cigar(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf03(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.3, 0.3, 0.4])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _bent_cigar(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _rosenbrock(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    # bi_rastrigin overwrites y[0:n]; it is the last component, as in the
    # reference, so no later component observes the corruption.
    total += _bi_rastrigin(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z, tmpx)
    return total


@njit(cache=True)
def _hf04(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.2, 0.2, 0.4])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _ellips(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _ackley(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _schaffer_f7(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf05(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.2, 0.3, 0.3])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _bent_cigar(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _hgbat(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _rosenbrock(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf06(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.2, 0.3, 0.3])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _escaffer6(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _hgbat(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _rosenbrock(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _schwefel(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf07(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.1, 0.2, 0.2, 0.2, 0.3])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _katsuura(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _ackley(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _grie_rosen(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _schwefel(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[4]:], g_nx[4], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf08(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _ellips(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _ackley(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _hgbat(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    total += _discus(y[g_off[4]:], g_nx[4], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf09(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _bent_cigar(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _grie_rosen(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _weierstrass(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    total += _escaffer6(y[g_off[4]:], g_nx[4], Os, Mr, 0, 0, y, z)
    return total


@njit(cache=True)
def _hf10(x, nx, Os, Mr, SS, s_flag, r_flag, y, z, tmpx, g_nx, g_off):
    gp = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    _hybrid_layout(gp, nx, g_nx, g_off)
    _hybrid_prepare(x, nx, Os, Mr, SS, s_flag, r_flag, y, z)
    total = _hgbat(y[g_off[0]:], g_nx[0], Os, Mr, 0, 0, y, z)
    total += _katsuura(y[g_off[1]:], g_nx[1], Os, Mr, 0, 0, y, z)
    total += _ackley(y[g_off[2]:], g_nx[2], Os, Mr, 0, 0, y, z)
    total += _rastrigin(y[g_off[3]:], g_nx[3], Os, Mr, 0, 0, y, z)
    total += _schwefel(y[g_off[4]:], g_nx[4], Os, Mr, 0, 0, y, z)
    total += _schaffer_f7(y[g_off[5]:], g_nx[5], Os, Mr, 0, 0, y, z)
    return total


# --------------------------------------------------------------------- #
# Composition helper (port of cf_cal)
# --------------------------------------------------------------------- #
@njit(cache=True)
def _cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    w = np.empty(cf_num)
    w_max = 0.0
    for i in range(cf_num):
        fit[i] += bias[i]
        acc = 0.0
        base = i * nx
        for j in range(nx):
            d = x[j] - Os[base + j]
            acc += d * d
        if acc != 0.0:
            w[i] = (1.0 / acc) ** 0.5 * np.exp(-acc / 2.0 / nx / (delta[i] ** 2))
        else:
            w[i] = INF
        if w[i] > w_max:
            w_max = w[i]
    w_sum = 0.0
    for i in range(cf_num):
        w_sum += w[i]
    if w_max == 0.0:
        for i in range(cf_num):
            w[i] = 1.0
        w_sum = cf_num
    total = 0.0
    for i in range(cf_num):
        total += w[i] / w_sum * fit[i]
    return total


# --------------------------------------------------------------------- #
# Dispatcher (port of cec17_test_func)
# --------------------------------------------------------------------- #
@njit(cache=True)
def cec17_evaluate(x, func_num, Os, Mr, SS, y, z, tmpx, g_nx, g_off, fit):
    nx = x.shape[0]
    nn = nx * nx
    if func_num == 1:
        return _bent_cigar(x, nx, Os, Mr, 1, 1, y, z) + 100.0
    if func_num == 2:
        return _sum_diff_pow(x, nx, Os, Mr, 1, 1, y, z) + 200.0
    if func_num == 3:
        return _zakharov(x, nx, Os, Mr, 1, 1, y, z) + 300.0
    if func_num == 4:
        return _rosenbrock(x, nx, Os, Mr, 1, 1, y, z) + 400.0
    if func_num == 5:
        return _rastrigin(x, nx, Os, Mr, 1, 1, y, z) + 500.0
    if func_num == 6:
        return _schaffer_f7(x, nx, Os, Mr, 1, 1, y, z) + 600.0
    if func_num == 7:
        return _bi_rastrigin(x, nx, Os, Mr, 1, 1, y, z, tmpx) + 700.0
    if func_num == 8:
        return _step_rastrigin(x, nx, Os, Mr, 1, 1, y, z) + 800.0
    if func_num == 9:
        return _levy(x, nx, Os, Mr, 1, 1, y, z) + 900.0
    if func_num == 10:
        return _schwefel(x, nx, Os, Mr, 1, 1, y, z) + 1000.0
    if func_num == 11:
        return _hf01(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1100.0
    if func_num == 12:
        return _hf02(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1200.0
    if func_num == 13:
        return _hf03(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1300.0
    if func_num == 14:
        return _hf04(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1400.0
    if func_num == 15:
        return _hf05(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1500.0
    if func_num == 16:
        return _hf06(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1600.0
    if func_num == 17:
        return _hf07(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1700.0
    if func_num == 18:
        return _hf08(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1800.0
    if func_num == 19:
        return _hf09(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 1900.0
    if func_num == 20:
        return _hf10(x, nx, Os, Mr, SS, 1, 1, y, z, tmpx, g_nx, g_off) + 2000.0

    if func_num == 21:  # cf01
        delta = np.array([10.0, 20.0, 30.0])
        bias = np.array([0.0, 100.0, 200.0])
        fit[0] = _rosenbrock(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[1] = _ellips(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 10000.0 * fit[1] / 1e10
        fit[2] = _rastrigin(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        return _cf_cal(x, nx, Os, delta, bias, fit, 3) + 2100.0
    if func_num == 22:  # cf02
        delta = np.array([10.0, 20.0, 30.0])
        bias = np.array([0.0, 100.0, 200.0])
        fit[0] = _rastrigin(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[1] = _griewank(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 1000.0 * fit[1] / 100.0
        fit[2] = _schwefel(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        return _cf_cal(x, nx, Os, delta, bias, fit, 3) + 2200.0
    if func_num == 23:  # cf03
        delta = np.array([10.0, 20.0, 30.0, 40.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0])
        fit[0] = _rosenbrock(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[1] = _ackley(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 1000.0 * fit[1] / 100.0
        fit[2] = _schwefel(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[3] = _rastrigin(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        return _cf_cal(x, nx, Os, delta, bias, fit, 4) + 2300.0
    if func_num == 24:  # cf04
        delta = np.array([10.0, 20.0, 30.0, 40.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0])
        fit[0] = _ackley(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[0] = 1000.0 * fit[0] / 100.0
        fit[1] = _ellips(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 10000.0 * fit[1] / 1e10
        fit[2] = _griewank(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[2] = 1000.0 * fit[2] / 100.0
        fit[3] = _rastrigin(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        return _cf_cal(x, nx, Os, delta, bias, fit, 4) + 2400.0
    if func_num == 25:  # cf05
        delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        fit[0] = _rastrigin(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[0] = 10000.0 * fit[0] / 1e3
        fit[1] = _happycat(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 1000.0 * fit[1] / 1e3
        fit[2] = _ackley(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[2] = 1000.0 * fit[2] / 100.0
        fit[3] = _discus(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        fit[3] = 10000.0 * fit[3] / 1e10
        fit[4] = _rosenbrock(x, nx, Os[4 * nx:], Mr[4 * nn:], 1, 1, y, z)
        return _cf_cal(x, nx, Os, delta, bias, fit, 5) + 2500.0
    if func_num == 26:  # cf06
        delta = np.array([10.0, 20.0, 20.0, 30.0, 40.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        fit[0] = _escaffer6(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[0] = 10000.0 * fit[0] / 2e7
        fit[1] = _schwefel(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[2] = _griewank(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[2] = 1000.0 * fit[2] / 100.0
        fit[3] = _rosenbrock(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        fit[4] = _rastrigin(x, nx, Os[4 * nx:], Mr[4 * nn:], 1, 1, y, z)
        fit[4] = 10000.0 * fit[4] / 1e3
        return _cf_cal(x, nx, Os, delta, bias, fit, 5) + 2600.0
    if func_num == 27:  # cf07
        delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        fit[0] = _hgbat(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[0] = 10000.0 * fit[0] / 1000.0
        fit[1] = _rastrigin(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 10000.0 * fit[1] / 1e3
        fit[2] = _schwefel(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[2] = 10000.0 * fit[2] / 4e3
        fit[3] = _bent_cigar(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        fit[3] = 10000.0 * fit[3] / 1e30
        fit[4] = _ellips(x, nx, Os[4 * nx:], Mr[4 * nn:], 1, 1, y, z)
        fit[4] = 10000.0 * fit[4] / 1e10
        fit[5] = _escaffer6(x, nx, Os[5 * nx:], Mr[5 * nn:], 1, 1, y, z)
        fit[5] = 10000.0 * fit[5] / 2e7
        return _cf_cal(x, nx, Os, delta, bias, fit, 6) + 2700.0
    if func_num == 28:  # cf08
        delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        fit[0] = _ackley(x, nx, Os[0:], Mr[0:], 1, 1, y, z)
        fit[0] = 1000.0 * fit[0] / 100.0
        fit[1] = _griewank(x, nx, Os[nx:], Mr[nn:], 1, 1, y, z)
        fit[1] = 1000.0 * fit[1] / 100.0
        fit[2] = _discus(x, nx, Os[2 * nx:], Mr[2 * nn:], 1, 1, y, z)
        fit[2] = 10000.0 * fit[2] / 1e10
        fit[3] = _rosenbrock(x, nx, Os[3 * nx:], Mr[3 * nn:], 1, 1, y, z)
        fit[4] = _happycat(x, nx, Os[4 * nx:], Mr[4 * nn:], 1, 1, y, z)
        fit[4] = 1000.0 * fit[4] / 1e3
        fit[5] = _escaffer6(x, nx, Os[5 * nx:], Mr[5 * nn:], 1, 1, y, z)
        fit[5] = 10000.0 * fit[5] / 2e7
        return _cf_cal(x, nx, Os, delta, bias, fit, 6) + 2800.0
    if func_num == 29:  # cf09
        delta = np.array([10.0, 30.0, 50.0])
        bias = np.array([0.0, 100.0, 200.0])
        fit[0] = _hf05(x, nx, Os[0:], Mr[0:], SS[0:], 1, 1, y, z, tmpx, g_nx, g_off)
        fit[1] = _hf06(x, nx, Os[nx:], Mr[nn:], SS[nx:], 1, 1, y, z, tmpx, g_nx, g_off)
        fit[2] = _hf07(x, nx, Os[2 * nx:], Mr[2 * nn:], SS[2 * nx:], 1, 1, y, z, tmpx, g_nx, g_off)
        return _cf_cal(x, nx, Os, delta, bias, fit, 3) + 2900.0
    if func_num == 30:  # cf10
        delta = np.array([10.0, 30.0, 50.0])
        bias = np.array([0.0, 100.0, 200.0])
        fit[0] = _hf05(x, nx, Os[0:], Mr[0:], SS[0:], 1, 1, y, z, tmpx, g_nx, g_off)
        fit[1] = _hf08(x, nx, Os[nx:], Mr[nn:], SS[nx:], 1, 1, y, z, tmpx, g_nx, g_off)
        fit[2] = _hf09(x, nx, Os[2 * nx:], Mr[2 * nn:], SS[2 * nx:], 1, 1, y, z, tmpx, g_nx, g_off)
        return _cf_cal(x, nx, Os, delta, bias, fit, 3) + 3000.0
    return np.nan


# --------------------------------------------------------------------- #
# Data loading and the public objective wrapper
# --------------------------------------------------------------------- #
def _read(name: str) -> np.ndarray:
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing official CEC2017 support file: {path}")
    cached = _CACHE_DIR / f"{name}.npy"
    if cached.exists():
        try:
            return np.load(cached)
        except Exception:  # a truncated cache file must never mask the source
            cached.unlink(missing_ok=True)
    data = np.genfromtxt(path, dtype=float)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _CACHE_DIR / f"{name}.{os.getpid()}.tmp.npy"
        np.save(temporary, data)
        os.replace(temporary, cached)
    except Exception:
        # The binary cache is an optimisation only; parsing the official text
        # file is always the fallback, so a read-only tree still works.
        pass
    return data


@lru_cache(maxsize=None)
def _load(func_num: int, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (OShift, M, SS) as flat arrays laid out like the reference.

    Cached per process.  The returned arrays are treated as read-only by every
    evaluation path (only the ``y``/``z``/``tmpx`` scratch buffers are written),
    so sharing one copy across every instance of a function is safe and avoids
    re-parsing multi-megabyte support files on every run.
    """
    matrix = np.ascontiguousarray(_read(f"M_{func_num}_D{dim}"), dtype=float)
    shift_raw = np.ascontiguousarray(_read(f"shift_data_{func_num}"), dtype=float)

    if func_num < 20:
        needed = dim * dim
        flat = matrix.reshape(-1)
        if flat.size < needed:
            raise ValueError(f"M_{func_num}_D{dim} has {flat.size} values, need {needed}")
        m_flat = flat[:needed].copy()
        o_flat = shift_raw.reshape(-1)[:dim].copy()
    else:
        # Hybrid 10 (F20) and every composition read a 10-block matrix file;
        # F20 uses only the first block, exactly as the reference does.
        flat = matrix.reshape(-1)
        m_flat = flat.copy()
        rows = np.atleast_2d(shift_raw)
        if rows.shape[0] == 1:
            o_flat = rows.reshape(-1)[:dim].copy()
        else:
            o_flat = np.ascontiguousarray(rows[:, :dim], dtype=float).reshape(-1).copy()

    if 11 <= func_num <= 20:
        ss = _read(f"shuffle_data_{func_num}_D{dim}").reshape(-1)[:dim]
    elif func_num in (29, 30):
        ss = _read(f"shuffle_data_{func_num}_D{dim}").reshape(-1)
    else:
        ss = np.ones(dim)
    return o_flat, m_flat, np.ascontiguousarray(ss, dtype=np.int64).copy()


CHARACTERISTICS: Dict[int, Tuple[str, str, str]] = {
    #  func: (group, modality, transform)
    1: ("unimodal", "unimodal", "shifted-rotated"),
    2: ("unimodal", "unimodal", "shifted-rotated"),
    3: ("unimodal", "unimodal", "shifted-rotated"),
    4: ("simple multimodal", "multimodal", "shifted-rotated"),
    5: ("simple multimodal", "multimodal", "shifted-rotated"),
    6: ("simple multimodal", "multimodal", "shifted"),
    7: ("simple multimodal", "multimodal", "shifted-rotated"),
    8: ("simple multimodal", "multimodal", "shifted-rotated"),
    9: ("simple multimodal", "multimodal", "shifted-rotated"),
    10: ("simple multimodal", "multimodal", "shifted-rotated"),
    11: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    12: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    13: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    14: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    15: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    16: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    17: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    18: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    19: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    20: ("hybrid", "multimodal", "shifted-rotated-shuffled"),
    21: ("composition", "multimodal", "composed"),
    22: ("composition", "multimodal", "composed"),
    23: ("composition", "multimodal", "composed"),
    24: ("composition", "multimodal", "composed"),
    25: ("composition", "multimodal", "composed"),
    26: ("composition", "multimodal", "composed"),
    27: ("composition", "multimodal", "composed"),
    28: ("composition", "multimodal", "composed"),
    29: ("composition", "multimodal", "composed-hybrid"),
    30: ("composition", "multimodal", "composed-hybrid"),
}

NAMES: Dict[int, str] = {
    1: "Shifted and Rotated Bent Cigar",
    2: "Shifted and Rotated Sum of Different Power (excluded)",
    3: "Shifted and Rotated Zakharov",
    4: "Shifted and Rotated Rosenbrock",
    5: "Shifted and Rotated Rastrigin",
    6: "Shifted and Rotated Expanded Schaffer F6/F7",
    7: "Shifted and Rotated Lunacek bi-Rastrigin",
    8: "Shifted and Rotated Non-Continuous Rastrigin",
    9: "Shifted and Rotated Levy",
    10: "Shifted and Rotated Schwefel",
    11: "Hybrid Function 1 (N=3)",
    12: "Hybrid Function 2 (N=3)",
    13: "Hybrid Function 3 (N=3)",
    14: "Hybrid Function 4 (N=4)",
    15: "Hybrid Function 5 (N=4)",
    16: "Hybrid Function 6 (N=4)",
    17: "Hybrid Function 7 (N=5)",
    18: "Hybrid Function 8 (N=5)",
    19: "Hybrid Function 9 (N=5)",
    20: "Hybrid Function 10 (N=6)",
    21: "Composition Function 1 (N=3)",
    22: "Composition Function 2 (N=3)",
    23: "Composition Function 3 (N=4)",
    24: "Composition Function 4 (N=4)",
    25: "Composition Function 5 (N=5)",
    26: "Composition Function 6 (N=5)",
    27: "Composition Function 7 (N=6)",
    28: "Composition Function 8 (N=6)",
    29: "Composition Function 9 (N=3)",
    30: "Composition Function 10 (N=3)",
}


@dataclass
class CEC2017Meta:
    """Metadata for one official CEC2017 instance."""

    name: str
    code: str
    func_num: int
    dim: int
    bounds: Tuple[float, float]
    optimum: float
    characteristic: str
    group: str
    modality: str
    transform: str
    shift_sha256: str
    matrix_sha256: str
    shuffle_sha256: str


class CEC2017Function:
    """Callable objective for one official CEC2017 function at one dimension."""

    __slots__ = ("func_num", "dim", "_os", "_m", "_ss", "_y", "_z", "_tmpx",
                 "_g_nx", "_g_off", "_fit", "_buf")

    def __init__(self, func_num: int, dim: int) -> None:
        if not 1 <= func_num <= 30:
            raise ValueError("func_num must be in 1..30")
        if dim not in SUPPORTED_DIMENSIONS:
            raise ValueError(f"dimension {dim} is not one of {SUPPORTED_DIMENSIONS}")
        if func_num in EXCLUDED_FUNCTIONS:
            # Still constructible so the validation harness can cover it.
            pass
        self.func_num = int(func_num)
        self.dim = int(dim)
        self._os, self._m, self._ss = _load(self.func_num, self.dim)
        self._y = np.zeros(dim, dtype=float)
        self._z = np.zeros(dim, dtype=float)
        self._tmpx = np.zeros(dim, dtype=float)
        self._g_nx = np.zeros(6, dtype=np.int64)
        self._g_off = np.zeros(6, dtype=np.int64)
        self._fit = np.zeros(6, dtype=float)
        self._buf = np.zeros(dim, dtype=float)

    def __call__(self, x) -> float:
        np.copyto(self._buf, np.asarray(x, dtype=float).reshape(-1))
        return float(
            cec17_evaluate(
                self._buf, self.func_num, self._os, self._m, self._ss,
                self._y, self._z, self._tmpx, self._g_nx, self._g_off, self._fit,
            )
        )

    @property
    def optimum(self) -> float:
        return 100.0 * self.func_num

    @property
    def optimum_position(self) -> np.ndarray:
        """The official shift vector o_i (first row for compositions)."""
        return self._os[: self.dim].copy()


@lru_cache(maxsize=None)
def _file_digest(name: str) -> str:
    """SHA-256 of one official support file, cached per process.

    The digests are what let the manifest prove which bytes were used, but the
    files total tens of megabytes, so hashing them once per process instead of
    once per suite construction matters when thousands of runs are enumerated.
    """
    if not name:
        return ""
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CEC2017Suite:
    """The official CEC 2017 suite at a fixed dimension.

    ``function_numbers`` defaults to the competition-usable set (F2 excluded).
    """

    def __init__(self, dim: int = 30, function_numbers: Sequence[int] | None = None,
                 with_digests: bool = True) -> None:
        self.dim = int(dim)
        self._with_digests = bool(with_digests)
        numbers = tuple(function_numbers) if function_numbers else EXPERIMENTAL_FUNCTIONS
        self.functions: Dict[str, Callable[[np.ndarray], float]] = {}
        self.meta: Dict[str, CEC2017Meta] = {}
        for func_num in numbers:
            code = f"F{func_num:02d}"
            objective = CEC2017Function(func_num, self.dim)
            group, modality, transform = CHARACTERISTICS[func_num]
            shuffle_name = (
                f"shuffle_data_{func_num}_D{self.dim}"
                if (11 <= func_num <= 20 or func_num in (29, 30))
                else ""
            )
            self.functions[code] = objective
            self.meta[code] = CEC2017Meta(
                name=f"CEC2017 {code}: {NAMES[func_num]}",
                code=code,
                func_num=func_num,
                dim=self.dim,
                bounds=(-100.0, 100.0),
                optimum=100.0 * func_num,
                characteristic=f"{group}; {modality}; {transform}",
                group=group,
                modality=modality,
                transform=transform,
                shift_sha256=_file_digest(f"shift_data_{func_num}") if self._with_digests else "",
                matrix_sha256=_file_digest(f"M_{func_num}_D{self.dim}") if self._with_digests else "",
                shuffle_sha256=_file_digest(shuffle_name) if (self._with_digests and shuffle_name) else "",
            )

    def get(self, code: str) -> Tuple[Callable[[np.ndarray], float], CEC2017Meta]:
        return self.functions[code], self.meta[code]

    def list_functions(self) -> List[str]:
        return list(self.functions.keys())


def data_provenance() -> Dict[str, object]:
    """Return the vendored-data provenance record."""
    path = DATA_DIR / "PROVENANCE.json"
    if not path.exists():
        return {"available": False}
    record = json.loads(path.read_text(encoding="utf-8"))
    record["available"] = True
    record["numba_enabled"] = bool(_HAVE_NUMBA) and os.environ.get("NUMBA_DISABLE_JIT") != "1"
    return record
