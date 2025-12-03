from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from landlab.core._validate import require_one_of
from landlab.core._validate import validate_array
from landlab.graph._neighbors import _neighbors_via_connector
from landlab.grid.base import ModelGrid


def _plural(at: str):
    if at == "patch":
        return "patches"
    else:
        return at + "s"


def map_neighbors(
    grid: ModelGrid,
    at: str,
    connector: str,
    *,
    where: ArrayLike[np.integer] | None = None,
    out: NDArray[np.integer] = None,
) -> NDArray[np.integer]:
    """Map neighboring grid elements via a connector.

    Parameters
    ----------
    grid : ModelGrid
        A Landlab grid that provides adjacency arrays such as
        ``faces_at_cell`` and ``cells_at_face``.
    at : {'node', 'link', 'patch', 'corner', 'face', 'cell'}
        Element type for which neighbors are requested.
    connector : {'link', 'face'}
        Connector type along which adjacency is computed.
    where : array_like of int, optional
        Which ``at`` elements to map. If not provided, map all elements.
    out : ndarray of int, optional
        Optional output array. If provided, must be same shape and dtype
        as the relevant ``<connectors>_at_<at>`` array, writable, and
        C-contiguous.

    Returns
    -------
    neighbors : ndarray of int
        Array of the same shape as ``<connectors>_at_<at>`` (e.g.,
        ``faces_at_cell``) where each entry gives the neighboring
        ``at`` element across the corresponding connector, or -1 if
        there is no valid neighbor.

    Examples
    --------
    >>> from landlab import RasterModelGrid

    Find neighboring patches of each patch for the following grid::

        o - o - o - o
        | 3 | 4 | 5 |
        o - o - o - o
        | 0 | 1 | 2 |
        o - o - o - o

    >>> grid = RasterModelGrid((3, 4))
    >>> map_neighbors(grid, at="patch", connector="link")
    array([[ 1,  3, -1, -1], [ 2,  4,  0, -1], [-1,  5,  1, -1],
           [ 4, -1, -1,  0], [ 5, -1,  3,  1], [-1, -1,  4,  2]])
    """
    connector = require_one_of(connector, allowed=("link", "face"))
    at = require_one_of(at, allowed=("node", "link", "patch", "corner", "face", "cell"))

    elements_at_connector_attr = f"{_plural(at)}_at_{connector}"
    try:
        elements_at_connector = getattr(grid, elements_at_connector_attr)
    except AttributeError as error:
        raise ValueError(
            f"unsupported mapping: grid is missing {elements_at_connector_attr}"
        ) from error

    if elements_at_connector.shape[1] != 2:
        raise ValueError(
            f"unsupported mapping: {elements_at_connector_attr} must be shape (n, 2)"
        )

    connectors_at_element_attr = f"{_plural(connector)}_at_{at}"
    try:
        connectors_at_element = getattr(grid, connectors_at_element_attr)
    except AttributeError as error:
        raise ValueError(
            f"unsupported mapping: grid is missing {connectors_at_element_attr}"
        ) from error

    if out is None:
        out = np.empty_like(connectors_at_element)

    if where is None:
        where = np.arange(
            connectors_at_element.shape[0], dtype=connectors_at_element.dtype
        )

    out = validate_array(
        out,
        dtype=connectors_at_element.dtype,
        shape=connectors_at_element.shape,
        writable=True,
        contiguous=True,
    )

    return _neighbors_via_connector(
        connectors_at_element, elements_at_connector, where=where, out=out
    )
