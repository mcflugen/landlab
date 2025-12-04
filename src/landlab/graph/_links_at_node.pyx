cimport cython
from libc.math cimport M_PI
from libc.math cimport atan2
from libc.stdint cimport int8_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdlib cimport free
from libc.stdlib cimport malloc

ctypedef fused id_t:
    int32_t
    int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _sort_links_at_node_by_angle(
    const id_t[:, ::1] nodes_at_link,
    id_t[:, ::1] links_at_node,
    int8_t[:, ::1] link_dirs_at_node,
    const double[::1] x_of_node,
    const double[::1] y_of_node,
):
    """Return a new links_at_node array with links sorted counterclockwise.

    Parameters
    ----------
    nodes_at_link : ndarray of int, shape (n_links, 2)
        Node identifiers (tail, head) for each link.
    links_at_node : ndarray of int, shape (n_nodes, max_links_per_node)
        Existing link IDs around each node. Unused entries must be -1.
    link_dirs_at_node : ndarray of int8, shape (n_nodes, max_links_per_node)
        Existing link directions around each node.
    x_of_node, y_of_node : ndarray of float, shape (n_nodes,)
        Coordinates of each node.
    """
    cdef Py_ssize_t n_nodes = links_at_node.shape[0]
    cdef Py_ssize_t max_links_per_node = links_at_node.shape[1]
    cdef Py_ssize_t k
    cdef Py_ssize_t node
    cdef id_t head
    cdef id_t link
    cdef id_t tail
    cdef double angle
    cdef double x0
    cdef double x1
    cdef double y0
    cdef double y1
    cdef double* angles = <double*> malloc(max_links_per_node * sizeof(double))
    cdef Py_ssize_t *n_links_at_node = <Py_ssize_t*>malloc(n_nodes * sizeof(Py_ssize_t))

    if angles == NULL or n_links_at_node == NULL:
        if angles != NULL:
            free(angles)
        if n_links_at_node != NULL:
            free(n_links_at_node)
        raise MemoryError("malloc failed in _sort_links_at_node_by_angle")

    try:
        with nogil:
            for node in range(n_nodes):
                n_links_at_node[node] = _compact_links(
                    &links_at_node[node, 0], &link_dirs_at_node[node, 0], max_links_per_node
                )

            for node in range(n_nodes):
                for k in range(n_links_at_node[node]):
                    link = links_at_node[node, k]

                    tail = nodes_at_link[link, 0]
                    head = nodes_at_link[link, 1]

                    x0 = x_of_node[node]
                    y0 = y_of_node[node]

                    if tail == node:
                        x1 = x_of_node[head]
                        y1 = y_of_node[head]
                    else:
                        x1 = x_of_node[tail]
                        y1 = y_of_node[tail]

                    angle = atan2(y1 - y0, x1 - x0)

                    if angle < 0.0:
                        angle += 2.0 * M_PI

                    angles[k] = angle

                _insertion_sort(
                    angles,
                    &links_at_node[node, 0],
                    &link_dirs_at_node[node, 0],
                    n_links_at_node[node],
                )
    finally:
        if angles != NULL:
            free(angles)
        if n_links_at_node != NULL:
            free(n_links_at_node)
