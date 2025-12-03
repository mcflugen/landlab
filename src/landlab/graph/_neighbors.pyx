cimport cython
from cython.parallel cimport prange
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t

ctypedef fused id_t:
    int32_t
    int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _neighbors_via_connector(
    const id_t[:, ::1] connectors_at_element,
    const id_t[:, ::1] elements_at_connector,
    const id_t[::1] where,
    id_t[:, ::1] out,
):
    """Find neighboring elements through connectors."""
    cdef Py_ssize_t n_y_per_x = connectors_at_element.shape[1]
    cdef Py_ssize_t i
    cdef Py_ssize_t k
    cdef Py_ssize_t x
    cdef Py_ssize_t y
    cdef id_t neighbor
    cdef id_t x0
    cdef id_t x1

    for i in prange(where.shape[0], nogil=True, schedule="static"):
        x = where[i]
        for k in range(n_y_per_x):
            y = connectors_at_element[x, k]
            if y < 0:
                out[x, k] = -1
                continue

            x0 = elements_at_connector[y, 0]
            x1 = elements_at_connector[y, 1]

            if x0 == x:
                neighbor = x1
            else:
                neighbor = x0

            out[x, k] = neighbor

    return (<object> out).base
