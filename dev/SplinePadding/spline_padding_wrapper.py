#---------------
from ctypes import CDLL
# from ctypes import POINTER
# from ctypes import c_double
from ctypes import c_int32

#---------------
import numpy as np

#---------------
from splineutilities import pole

lib = CDLL("./splinepadding.dylib")
lib.get_samples_to_coeff_p.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_n.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_w.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_a.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_np.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_nn.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
lib.get_samples_to_coeff_nw.argtypes = (
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32,
    np.ctypeslib.ndpointer(dtype = float, ndim = 1),
    c_int32
)
# lib.get_samples_to_coeff.restype = POINTER(c_double)

def samples_to_coeff_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if (1 >= degree) or (1 == K):
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_p(data, K, poles, len(poles))

def samples_to_coeff_n (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if (1 >= degree) or (1 == K):
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_n(data, K, poles, len(poles))

def samples_to_coeff_w (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if (1 >= degree) or (1 == K):
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_w(data, K, poles, len(poles))

def samples_to_coeff_a (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if (1 >= degree) or (1 == K):
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_a(data, K, poles, len(poles))

def samples_to_coeff_np (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if 1 >= degree:
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_np(data, K, poles, len(poles))

def samples_to_coeff_nn (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if 1 >= degree:
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_nn(data, K, poles, len(poles))

def samples_to_coeff_nw (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:
    K = len(data)
    if 1 >= degree:
        return
    data = np.ascontiguousarray(data, dtype = float)
    poles = pole(degree)
    lib.get_samples_to_coeff_nw(data, K, poles, len(poles))














import splinepadding
import timeit
import copy
import math

max_degree = 20
length_factor = max_degree + 1 + 4

[pole(degree) for degree in range(max_degree + 1)]
f = [
    [
        np.ascontiguousarray(
            np.random.default_rng().normal(0.0, 1.0, size = K),
            dtype = float
        )
        for K in range(1, length_factor * (n + 1) + 1)
    ]
    for n in range(max_degree + 1)
]
e = copy.deepcopy(f)

#---------------
def ep():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_p(f1, degree = n)
            splinepadding.samples_to_coeff_p(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def en():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_n(f1, degree = n)
            splinepadding.samples_to_coeff_n(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def ew():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_w(f1, degree = n)
            splinepadding.samples_to_coeff_w(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def ea():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_a(f1, degree = n)
            splinepadding.samples_to_coeff_a(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def enp():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_np(f1, degree = n)
            splinepadding.samples_to_coeff_np(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def enn():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_nn(f1, degree = n)
            splinepadding.samples_to_coeff_nn(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )
#---------------
def enw():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            f1 = f[n][K - 1].copy()
            f2 = f[n][K - 1].copy()
            samples_to_coeff_nw(f1, degree = n)
            splinepadding.samples_to_coeff_nw(f2, degree = n)
            e[n][K - 1] = math.fsum(
                abs(f1[k] - f2[k])
                for k in range(len(f[n][K - 1]))
            )

#---------------
def swp():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_p(f0[n][K - 1], degree = n)
#---------------
def pyp():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_p(f0[n][K - 1], degree = n)
#---------------
def swn():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_n(f0[n][K - 1], degree = n)
#---------------
def pyn():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_n(f0[n][K - 1], degree = n)
#---------------
def sww():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_w(f0[n][K - 1], degree = n)
#---------------
def pyw():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_w(f0[n][K - 1], degree = n)
#---------------
def swa():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_a(f0[n][K - 1], degree = n)
#---------------
def pya():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_a(f0[n][K - 1], degree = n)
#---------------
def swnp():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_np(f0[n][K - 1], degree = n)
#---------------
def pynp():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_np(f0[n][K - 1], degree = n)
#---------------
def swnn():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_nn(f0[n][K - 1], degree = n)
#---------------
def pynn():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_nn(f0[n][K - 1], degree = n)
#---------------
def swnw():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            samples_to_coeff_nw(f0[n][K - 1], degree = n)
#---------------
def pynw():
    for n in range(max_degree + 1):
        for K in range(1, length_factor * (n + 1) + 1):
            splinepadding.samples_to_coeff_nw(f0[n][K - 1], degree = n)

e = copy.deepcopy(f)
ep()
print("difference: p = {}".format(e))
e = copy.deepcopy(f)
en()
print("difference: n = {}".format(e))
e = copy.deepcopy(f)
ew()
print("difference: w = {}".format(e))
e = copy.deepcopy(f)
ea()
print("difference: a = {}".format(e))
e = copy.deepcopy(f)
enp()
print("difference: np = {}".format(e))
e = copy.deepcopy(f)
enn()
print("difference: nn = {}".format(e))
e = copy.deepcopy(f)
enw()
print("difference: nw = {}".format(e))

f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pyp())
print("python: p = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swp())
print("swift : p = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pyn())
print("python: n = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swn())
print("swift : n = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pyw())
print("python: w = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: sww())
print("swift : w = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pya())
print("python: a = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swa())
print("swift : a = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pynp())
print("python: np = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swnp())
print("swift : np = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pynn())
print("python: nn = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swnn())
print("swift : nn = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: pynw())
print("python: nw = {}".format(timer.timeit(1)))
f0 = copy.deepcopy(f)
timer = timeit.Timer(lambda: swnw())
print("swift : nw = {}".format(timer.timeit(1)))
