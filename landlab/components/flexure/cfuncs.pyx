from multiprocessing import Pool

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs
from libc.stdlib cimport abs


DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(True)
def subside_parallel_row(
    np.ndarray[DTYPE_t, ndim=1] w,
    np.ndarray[DTYPE_t, ndim=1] load,
    np.ndarray[DTYPE_t, ndim=1] r,
    DTYPE_t alpha,
    DTYPE_t gamma_mantle
):
  cdef long ncols = w.size
  cdef double inv_c = 1. / (2. * np.pi * gamma_mantle * alpha ** 2.)
  cdef double c
  cdef long col_load
  cdef long col

  return

  for col_load in range(ncols):
    if fabs(load[col_load]) > 1e-6:
      c = load[col_load] * inv_c
      # for j in range(ncols):
      #   w[j] += - c * r[abs(j - i)]

      for col in range(col_load):
        w[col] += - c * r[col_load - col]
      for col in range(col_load, ncols):
        w[col] += - c * r[col - col_load]


@cython.boundscheck(True)
def subside_grid(
    np.ndarray[DTYPE_t, ndim=2] w,
    np.ndarray[DTYPE_t, ndim=2] load,
    np.ndarray[DTYPE_t, ndim=2] r,
    DTYPE_t alpha,
    DTYPE_t gamma_mantle
):
  cdef long nrows = w.shape[0]
  cdef long row_load
  cdef long row

  for row_load in range(nrows):
    # for j in range(nrows):
    #   subside_parallel_row(w[j], load[i], r[abs(j - i)], alpha, gamma_mantle)

    for row in range(row_load):
      subside_parallel_row(w[row], load[row_load], r[row_load - row], alpha, gamma_mantle)
    for row in range(row_load, nrows):
      subside_parallel_row(w[row], load[row_load], r[row - row_load], alpha, gamma_mantle)


def subside_grid_strip(
    np.ndarray[DTYPE_t, ndim=2] load,
    np.ndarray[DTYPE_t, ndim=2] r,
    DTYPE_t alpha,
    DTYPE_t gamma_mantle,
    strip_range
):
  (start, stop) = strip_range

  cdef np.ndarray w = np.zeros((stop - start, load.shape[1]), dtype=DTYPE)
  cdef i
  cdef j
  cdef nrows = load.shape[0]

  for i in range(nrows):
    for j in range(start, stop):
      subside_parallel_row(w[j - start], load[i], r[abs(j - i)], alpha, gamma_mantle)

  return w, strip_range


def tile_grid_into_strips(grid, n_strips):
    rows_per_strip = grid.shape[0] // n_strips

    starts = np.arange(0, grid.shape[0], rows_per_strip)
    stops = starts + rows_per_strip
    stops[-1] = grid.shape[0]

    return zip(starts, stops)


def _subside_grid_strip_helper(args):
  return subside_grid_strip(*args)


def subside_grid_in_parallel(
    np.ndarray[DTYPE_t, ndim=2] w,
    np.ndarray[DTYPE_t, ndim=2] load,
    np.ndarray[DTYPE_t, ndim=2] r,
    DTYPE_t alpha,
    DTYPE_t gamma_mantle,
    n_procs
):
    if n_procs == 1:
        return subside_grid(w, load, r, alpha, gamma_mantle)

    strips = tile_grid_into_strips(w, n_procs)

    args = [(load, r, alpha, gamma_mantle, strip) for strip in strips]

    pool = Pool(processes=n_procs)

    results = pool.map(_subside_grid_strip_helper, args)
    for dz, strip in results:
        start, stop = strip
        w[start:stop] += dz
