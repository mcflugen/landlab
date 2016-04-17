#! /usr/bin/env python
"""Calculate vector divergence and related quantities at nodes or cells."""
import numpy as np
from landlab.utils.decorators import use_field_name_or_array


@use_field_name_or_array('face')
def calc_net_face_flux_at_cells(grid, unit_flux_at_faces, out=None):
    """Calculate net face fluxes at cells.

    Given a flux per unit width across each face in the grid, calculate the net
    outflux (or influx, if negative) at each cell.

    Construction::

        calc_net_face_flux_at_cells(grid, unit_flux_at_faces, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    unit_flux_at_faces : ndarray or field name
        Flux per unit width associated with faces.
    out : ndarray, optional
        Buffer to hold the result.

    Returns
    -------
    ndarray (x number of cells)
        Net flux at cells.

    Examples
    --------
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
    >>> rg = RasterModelGrid(3, 4, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation')
    >>> z[5] = 50.0
    >>> z[6] = 36.0
    >>> fg = rg.calculate_gradients_at_faces(z)  # there are 7 faces
    >>> fg
    array([ 5. ,  3.6,  5. , -1.4, -3.6, -5. , -3.6])
    >>> calc_net_face_flux_at_cells(rg, -fg)
    array([-164.,  -94.])
    >>> rg.set_status_at_node_on_edges(right=CLOSED_BOUNDARY)
    >>> rg.set_status_at_node_on_edges(top=CLOSED_BOUNDARY)
    >>> unit_flux_at_faces = np.zeros(rg.number_of_faces)
    >>> unit_flux_at_faces[rg.active_faces] = -fg[rg.active_faces]
    >>> calc_net_face_flux_at_cells(rg, unit_flux_at_faces)
    array([-114.,  -22.])

    >>> from landlab import HexModelGrid
    >>> hg = HexModelGrid(3, 3, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation', noclobber=False)
    >>> z[4] = 50.0
    >>> z[5] = 36.0
    >>> fg = hg.calculate_gradients_at_faces(z)  # there are 11 faces
    >>> fg
    array([ 5. ,  5. ,  3.6,  3.6,  5. , -1.4, -3.6, -5. , -5. , -3.6, -3.6])
    >>> nffc = calc_net_face_flux_at_cells(hg, -fg)
    >>> np.round(nffc)
    array([-152.,  -96.])

    Notes
    -----
    This is essentially a line integral for the fluxes along the boundaries of
    each cell. Hence, the resulting output has dimensions of total flux (so,
    if the unit flux happens to be mass per time per face width, the output
    will be in mass per unit time).
    """
    if out is None:
        out = grid.empty(centering='cell')
    total_flux = unit_flux_at_faces * grid.face_width
    out = np.zeros(grid.number_of_cells)
    fac = grid.faces_at_cell
    for c in range(grid.link_dirs_at_node.shape[1]):
        out += total_flux[fac[:,c]] \
        * grid.link_dirs_at_node[grid.node_at_cell,c]
    return out


@use_field_name_or_array('face')
def calc_face_flux_divergence_at_cells(grid, unit_flux_at_faces):
    """Calculate divergence of face-based fluxes at cells.

    Given a flux per unit width across each face in the grid, calculate the net
    outflux (or influx, if negative) divided by cell area, at each cell.

    Construction::

        calc_face_flux_divergence_at_cells(grid, unit_flux_at_faces, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    unit_flux_at_faces : ndarray or field name
        Flux per unit width associated with faces.

    Returns
    -------
    ndarray (x number of cells)
        Flux divergence at cells.

    Examples
    --------
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
    >>> rg = RasterModelGrid(3, 4, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation')
    >>> z[5] = 50.0
    >>> z[6] = 36.0
    >>> fg = rg.calculate_gradients_at_faces(z)  # there are 7 faces
    >>> fg
    array([ 5. ,  3.6,  5. , -1.4, -3.6, -5. , -3.6])
    >>> calc_face_flux_divergence_at_cells(rg, -fg)
    array([-1.64, -0.94])
    >>> rg.set_status_at_node_on_edges(right=CLOSED_BOUNDARY)
    >>> rg.set_status_at_node_on_edges(top=CLOSED_BOUNDARY)
    >>> unit_flux_at_faces = np.zeros(rg.number_of_faces)
    >>> unit_flux_at_faces[rg.active_faces] = -fg[rg.active_faces]
    >>> calc_face_flux_divergence_at_cells(rg, unit_flux_at_faces)
    array([-1.14, -0.22])

    Notes
    -----
    Performs a numerical flux divergence operation on cells.
    """
    return calc_net_face_flux_at_cells(grid, unit_flux_at_faces) \
           / grid.area_of_cell


@use_field_name_or_array('link')
def calc_net_link_flux_at_nodes(grid, unit_flux_at_links, out=None):
    """Calculate net link fluxes at nodes.

    Given a flux per unit width along each link in the grid, calculate the net
    outflux (or influx, if negative) at each node. Fluxes are treated as zero
    for links that have no faces, and net fluxes are treated as zero for nodes
    that have no cell.

    Construction::

        calc_net_link_flux_at_nodes(grid, unit_flux_at_links, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    unit_flux_at_links : ndarray or field name
        Flux per unit width associated with links.
    out : ndarray, optional
        Buffer to hold the result.

    Returns
    -------
    ndarray (x number of cells)
        Net flux at nodes.

    Examples
    --------
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
    >>> rg = RasterModelGrid(3, 4, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation')
    >>> z[5] = 50.0
    >>> z[6] = 36.0
    >>> lg = rg.calculate_gradients_at_links(z)  # there are 17 links
    >>> lg
    array([ 0. ,  0. ,  0. ,  0. ,  5. ,  3.6,  0. ,  5. , -1.4, -3.6,  0. ,
           -5. , -3.6,  0. ,  0. ,  0. ,  0. ])
    >>> calc_net_link_flux_at_nodes(rg, -lg)
    array([   0.,    0.,    0.,    0.,    0., -164.,  -94.,    0.,    0.,
              0.,    0.,    0.])
    >>> rg.set_status_at_node_on_edges(right=CLOSED_BOUNDARY)
    >>> rg.set_status_at_node_on_edges(top=CLOSED_BOUNDARY)
    >>> unit_flux_at_links = np.zeros(rg.number_of_links)
    >>> unit_flux_at_links[rg.active_links] = -lg[rg.active_links]
    >>> nlfn = calc_net_link_flux_at_nodes(rg, unit_flux_at_links)
    >>> np.round(nlfn)
    array([   0.,    0.,    0.,    0.,    0., -114.,  -22.,    0.,    0.,
              0.,    0.,    0.])

    >>> from landlab import HexModelGrid
    >>> hg = HexModelGrid(3, 3, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation', noclobber=False)
    >>> z[4] = 50.0
    >>> z[5] = 36.0
    >>> lg = hg.calculate_gradients_at_links(z)  # there are ? links
    >>> lg
    array([ 0. ,  0. ,  0. ,  5. ,  5. ,  3.6,  3.6,  0. ,  5. , -1.4, -3.6,
            0. , -5. , -5. , -3.6, -3.6,  0. ,  0. ,  0. ])
    >>> nlfn = calc_net_link_flux_at_nodes(hg, -lg)
    >>> np.round(nlfn)
    array([   0.,    0.,    0.,    0., -152.,  -96.,    0.,    0.,    0.,    0.])

    Notes
    -----
    This is essentially a line integral for the fluxes along the boundaries of
    each cell. Hence, the resulting output has dimensions of total flux (so,
    if the unit flux happens to be mass per time per face width, the output
    will be in mass per unit time). Because a line integral is undefined where
    there are no cells (i.e., perimeter nodes), the result is given as zeros
    for these nodes. The current algorithm uses fancy indexing (calling
    calc_net_face_flux_at_cells) and could probably be made faster.
    """
    if out is None:
        out = grid.zeros(centering='node')

    out[grid.node_at_cell] = calc_net_face_flux_at_cells(grid, 
                                unit_flux_at_links[grid.link_at_face])
    return out


@use_field_name_or_array('link')
def calc_link_flux_divergence_at_nodes(grid, unit_flux_at_links, out=None):
    """Calculate divergence of link-based fluxes at nodes.

    Given a flux per unit width across each face in the grid, calculate the net
    outflux (or influx, if negative) divided by cell area, at each node (zero
    or "out" value for nodes without cells).

    Construction::

        calc_link_flux_divergence_at_nodes(grid, unit_flux_at_links, out=None)

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    unit_flux_at_links : ndarray or field name
        Flux per unit width associated with links.

    Returns
    -------
    ndarray (x number of nodes)
        Flux divergence at nodes.

    Examples
    --------
    >>> from landlab import RasterModelGrid, CLOSED_BOUNDARY
    >>> rg = RasterModelGrid(3, 4, 10.0)
    >>> z = rg.add_zeros('node', 'topographic__elevation')
    >>> z[5] = 50.0
    >>> z[6] = 36.0
    >>> lg = rg.calculate_gradients_at_links(z)  # there are 17 links
    >>> lg
    array([ 0. ,  0. ,  0. ,  0. ,  5. ,  3.6,  0. ,  5. , -1.4, -3.6,  0. ,
           -5. , -3.6,  0. ,  0. ,  0. ,  0. ])
    >>> calc_link_flux_divergence_at_nodes(rg, -lg)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.64, -0.94,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ])
    >>> rg.set_status_at_node_on_edges(right=CLOSED_BOUNDARY)
    >>> rg.set_status_at_node_on_edges(top=CLOSED_BOUNDARY)
    >>> unit_flux_at_links = np.zeros(rg.number_of_links)
    >>> unit_flux_at_links[rg.active_links] = -lg[rg.active_links]
    >>> calc_link_flux_divergence_at_nodes(rg, unit_flux_at_links)
    array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.14, -0.22,  0.  ,  0.  ,
            0.  ,  0.  ,  0.  ])

    Notes
    -----
    Performs a numerical flux divergence operation on nodes.
    """
    if out is None:
        out = grid.zeros(centering='node')
    
    out[grid.node_at_cell] = calc_net_face_flux_at_cells(grid, 
                                unit_flux_at_links[grid.link_at_face]) \
                                / grid.area_of_cell
    return out

