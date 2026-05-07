import warnings

import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from skimage import measure

def load_bmesh(bmesh):
    """
    Transforms the `bmesh` parameter into a standardized dictionary of PyVista PolyData meshes.

    Parameters
    ----------
    bmesh : None, str, dict, or pyvista.PolyData
        - None: Returns an empty dictionary (disables background mesh).
        - str: Fetches standard meshes from the registry (e.g., 'midthickness').
        - dict: Maps custom meshes to 'L' and 'R' keys. Values can be pre-loaded 
          PyVista PolyData objects or string file paths (which are auto-loaded).
        - pyvista.PolyData: A single unified mesh, mapped to the 'both' key.

    Returns
    -------
    dict
        Standardized dictionary containing the loaded PyVista meshes.
    """
    from .data import get_surface_paths
    from .utils import load_gii2pv

    if bmesh is None:
        return {}
    if isinstance(bmesh, str):
        lh_path, rh_path = get_surface_paths(bmesh, 'bmesh')
        return {'L': load_gii2pv(lh_path), 'R': load_gii2pv(rh_path)}
    if isinstance(bmesh, dict):
        clean_dict = {}
        for k, v in bmesh.items():
            if isinstance(v, str):
                if v.endswith('.gii') or v.endswith('.gii.gz'):
                    v = load_gii2pv(v)
                else:
                    v = pv.read(v)
            if k.upper() in ['L', 'LEFT']: clean_dict['L'] = v
            elif k.upper() in ['R', 'RIGHT']: clean_dict['R'] = v
            else: clean_dict[k] = v
        return clean_dict
    
    return {'both': bmesh}

def extract_polydata(mesh_hemi: pv.PolyData):
    """Return vertices and rotated faces for plotting."""
    v = mesh_hemi.points
    f = mesh_hemi.faces.reshape(-1, 4)[:, 1:]
    return v, f
    
def make_cortical_mesh(verts, faces, scalars, scalar_name='Data'):
    """
    Converts standard triangle face arrays into pyvista's specific padded format 
    and injects per-vertex data.
    
    Parameters
    ----------
    verts : numpy.ndarray
        (N, 3) float array of spatial vertex coordinates (x, y, z).
    faces : numpy.ndarray
        (M, 3) int array of triangle face indices.
    scalars : numpy.ndarray
        (N,) float array of per-vertex scalar values.
    scalar_name : str, optional
        the string key to store the data under. default is 'Data'.
        
    Returns
    -------
    mesh : pyvista.PolyData
        the instantiated pyvista mesh with attached scalar data.
    """
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten().astype(int)
    mesh = pv.PolyData(verts, faces_pv)
    mesh[scalar_name] = scalars
    return mesh

def load_vertexwise_mesh(lh_mesh_path, rh_mesh_path, lh_data, rh_data, scalar_name='Data'):
    """
    Loads GIfTI geometry files (i.e. brain mesh), converts them to pyvista meshes, and injects 
    the provided 1D data arrays into them.
    
    Parameters
    ----------
    lh_mesh_path : str
        absolute path to the left hemisphere geometry file (e.g., .surf.gii).
    rh_mesh_path : str
        absolute path to the right hemisphere geometry file (e.g., .surf.gii).
    lh_data : numpy.ndarray
        1D array of scalar values for the left hemisphere vertices.
    rh_data : numpy.ndarray
        1D array of scalar values for the right hemisphere vertices.
    scalar_name : str, optional
        the string key to store the data under in the pyvista point data dictionary. 
        default is 'Data'.
        
    Returns
    -------
    lh_mesh, rh_mesh : tuple of pyvista.PolyData
        left and right hemisphere meshes ready for `yabplot.plotting.plot_vertexwise`.
    """
    from .utils import load_gii
    lh = make_cortical_mesh(*load_gii(lh_mesh_path), lh_data, scalar_name)
    rh = make_cortical_mesh(*load_gii(rh_mesh_path), rh_data, scalar_name)
    return lh, rh

def load_nii_as_mesh(
    nii_path,
    threshold=0.5,
    blur_sigma=1.5,
    smooth_i=10,
    smooth_f=0.1
):
    """
    Build a surface mesh from a 3D NIfTI volume using marching cubes, with optional Gaussian blurring and mesh smoothing.

    Parameters
    ----------
    nii_path : str
        Absolute path to a NIfTI file representing a 3D volume. If 4D, only the first volume will be used.
    threshold : float, optional
        Threshold applied after optional blur. Voxels ``> threshold`` are kept.
    blur_sigma : float, optional
        Gaussian blur (voxel units) before thresholding.
    smooth_i : int, optional
        Number of PyVista smoothing iterations after surface extraction.
    smooth_f : float, optional
        Relaxation factor for mesh smoothing.

    Returns
    -------
    mesh : pyvista.PolyData
        The extracted and smoothed surface mesh ready for plotting.
    """

    img = nib.load(nii_path)
    vol = img.get_fdata()

    if vol.ndim > 3:
        warnings.warn(
            f"[WARNING] detected {vol.ndim}d nifti volume. using the first volume (index 0)."
        )
        vol = vol[..., 0]

    vol = np.nan_to_num(vol, nan=0.0)

    if blur_sigma and blur_sigma > 0:
        vol = gaussian_filter(vol, sigma=float(blur_sigma))

    mask = vol > float(threshold)

    if not np.any(mask):
        raise ValueError("Mask is empty after thresholding. Adjust threshold/blur_sigma.")

    verts_vox, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    verts_world = nib.affines.apply_affine(img.affine, verts_vox)

    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces.astype(np.int64)
    ]).ravel()
    mesh = pv.PolyData(verts_world.astype(np.float32), faces_pv)

    if smooth_i and smooth_i > 0:
        mesh = mesh.smooth(n_iter=int(smooth_i), relaxation_factor=float(smooth_f))

    if mesh.n_points == 0:
        raise ValueError("Extracted mesh has no vertices. Check input mask and parameters.")

    # fill topological holes in the extracted meshes
    try:
        mesh = mesh.fill_holes(1000)
    except Exception as e:
        warnings.warn(f"Mesh hole filling failed: {e}. Continuing with unfilled meshes.")

    return mesh

def map_values_to_surface(data, target_labels, lut_ids, dense_lut_names):
    """maps data to vertices."""
    # filter valid regions
    valid_ids_list = []
    valid_names_list = []
    
    for rid in lut_ids:
        if rid < len(dense_lut_names):
            valid_ids_list.append(rid)
            valid_names_list.append(dense_lut_names[rid])
    
    valid_ids = np.array(valid_ids_list)
    n_regions = len(valid_ids)

    # atlas visualization without data
    if data is None:
        return target_labels

    # data mapping
    max_id = max(target_labels.max(), lut_ids.max())
    lookup_table = np.full(max_id + 1, np.nan)
    source_values = np.full(n_regions, np.nan)

    if isinstance(data, dict):
        for i, name in enumerate(valid_names_list):
            if name in data:
                source_values[i] = data[name]            
    elif isinstance(data, (np.ndarray, list, tuple)):
        # map by order
        if len(data) != n_regions:
            raise ValueError(
                f"Data length mismatch! The atlas LUT defines {n_regions} regions, "
                f"but input data has {len(data)}.\n"
                f"Expected order starts with: {valid_names_list[0:3]}...\n"
                f"Solution: Use a dictionary for partial data, or check `yabplot.get_atlas_regions`."
            )
        source_values = np.array(data)
    else:
        raise ValueError("Data must be dict, list, or numpy array.")

    lookup_table[valid_ids] = source_values
    return lookup_table[target_labels]

def get_adj(faces, n_v):
    """build adjacency matrix from faces."""
    row, col = [], []
    for tri in faces:
        row.extend([tri[0], tri[1], tri[2], tri[0], tri[1], tri[2]])
        col.extend([tri[1], tri[2], tri[0], tri[2], tri[0], tri[1]])
    adj = sp.csc_matrix((np.ones_like(row), (row, col)), shape=(n_v, n_v))
    adj.data = np.ones_like(adj.data)
    return adj

def get_smooth_mask(faces, data, iterations=4):
    """blur binary mask for guide of geometric slicing."""
    n_v = len(data)
    mask = data.astype(np.float64)
    adj = get_adj(faces, n_v)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0 
    for _ in range(iterations):
        mask = (mask + (adj.dot(mask) / deg)) / 2.0
    return mask

def apply_internal_blur(faces, data, iterations=1, weight=0.2):
    """blur data only on borders where different regions touch."""
    data_out = np.copy(data)
    n_v = len(data)
    adj = get_adj(faces, n_v)
    rows, cols = adj.nonzero()
    valid = ~np.isnan(data_out)
    diff = valid[rows] & valid[cols] & ~np.isclose(data_out[rows], data_out[cols], atol=1e-5)
    b_verts = np.unique(rows[diff])
    
    if len(b_verts) == 0: return data_out

    for _ in range(iterations):
        temp = np.nan_to_num(data_out, nan=0.0)
        v_counts = adj.dot(valid.astype(float))
        v_counts[v_counts == 0] = 1.0
        n_mean = adj.dot(temp) / v_counts
        data_out[b_verts] = (1 - weight) * data_out[b_verts] + weight * n_mean[b_verts]
    return data_out

def apply_dilation(faces, data, iterations=4):
    """push values into NaN space to keep geometric cut pure."""
    data_out = np.copy(data)
    n_v = len(data)
    adj = get_adj(faces, n_v)
    for _ in range(iterations):
        nan_m = np.isnan(data_out)
        temp = np.nan_to_num(data_out, nan=0.0)
        v_counts = adj.dot((~nan_m).astype(float))
        s_neighbors = adj.dot(temp)
        u_mask = nan_m & (v_counts > 0)
        data_out[u_mask] = s_neighbors[u_mask] / v_counts[u_mask]
    return data_out


def get_puzzle_pieces(v, f, raw_vals):
    """carve out geometric pieces with slight overlap to prevent gaps."""
    pieces = []
    valid_mask = ~np.isnan(raw_vals) & (raw_vals != 0.0)
    u_vals = np.unique(raw_vals[valid_mask])
    master = make_cortical_mesh(v, f, np.zeros_like(raw_vals))

    for val in u_vals:
        r_mask = np.where(raw_vals == val, 1.0, 0.0)
        s_mask = get_smooth_mask(f, r_mask, iterations=4)
        temp = master.copy()
        temp['Slice_Mask'] = s_mask
        # reduce search space
        patch = temp.threshold(0.01, scalars='Slice_Mask')
        if patch.n_points > 0:
            # use 0.48 (slightly expanded) for pieces to seal cracks
            piece = patch.clip_scalar(scalars='Slice_Mask', value=0.48, invert=False)
            if piece.n_points > 0:
                piece['Data'] = np.full(piece.n_points, val)
                pieces.append(piece)
    
    # slice base brain
    all_mask = np.where(valid_mask, 1.0, 0.0)
    s_all = get_smooth_mask(f, all_mask, iterations=4)
    master['Slice_Mask'] = s_all
    # use 0.52 (slightly contracted) for the hole to ensure colored pieces cover the edge
    base_p = master.clip_scalar(scalars='Slice_Mask', value=0.52, invert=True)
    if base_p.n_points > 0:
        base_p['Data'] = np.full(base_p.n_points, np.nan)
    
    return base_p, pieces


def get_region_boundaries(vertices, faces, labels, values=None, include_nan=True,
                           region_ids=None):
    """Extract boundary edges between adjacent regions as a PyVista PolyData.

    An edge is a boundary if its two endpoint vertices carry different label IDs.

    Parameters
    ----------
    vertices : numpy.ndarray
        (N, 3) float array of vertex coordinates.
    faces : numpy.ndarray
        (M, 3) int array of triangle face indices.
    labels : numpy.ndarray
        (N,) integer label array (one ID per vertex).
    values : numpy.ndarray, optional
        (N,) float array of mapped data values (may contain NaN). When provided
        together with ``include_nan=False``, edges where either endpoint has a
        NaN value are excluded.
    include_nan : bool, optional
        If False and ``values`` is provided, suppress contour edges where
        *both* endpoints are NaN — i.e. draw outlines around regions that have
        data (including their border with NaN regions) but skip borders between
        two NaN regions. Default True.
    region_ids : set or array-like of int, optional
        If provided, only boundary edges where at least one endpoint vertex
        belongs to one of these label IDs are included. Draws the full outline
        of each selected region (including its border with unselected regions).

    Returns
    -------
    pyvista.PolyData or None
        Line mesh of boundary edges, or None if no boundaries exist.
    """
    # collect all unique undirected edges from the triangle mesh
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # keep only edges where the two endpoints belong to different regions
    with np.errstate(invalid='ignore'):
        boundary_mask = labels[edges[:, 0]] != labels[edges[:, 1]]

    # optionally restrict to edges touching the requested region IDs
    if region_ids is not None:
        region_arr = np.asarray(list(region_ids))
        region_mask = (np.isin(labels[edges[:, 0]], region_arr) |
                       np.isin(labels[edges[:, 1]], region_arr))
        boundary_mask = boundary_mask & region_mask

    # optionally suppress edges that touch NaN-valued vertices
    if not include_nan and values is not None:
        v0 = values[edges[:, 0]]
        v1 = values[edges[:, 1]]
        with np.errstate(invalid='ignore'):
            # keep edge if at least one side has data — draws the full outline
            # around regions with values, including their border with NaN regions
            valid_mask = ~np.isnan(v0.astype(float)) | ~np.isnan(v1.astype(float))
        boundary_mask = boundary_mask & valid_mask

    boundary_edges = edges[boundary_mask]

    if len(boundary_edges) == 0:
        return None

    lines = np.column_stack([
        np.full(len(boundary_edges), 2, dtype=np.int64),
        boundary_edges.astype(np.int64)
    ]).ravel()

    return pv.PolyData(vertices, lines=lines)


def lines_from_streamlines(streamlines):
    if len(streamlines) == 0: return np.array([]), np.array([]), np.array([])
    
    points = np.vstack(streamlines)
    n_points = [len(s) for s in streamlines]
    offsets = np.insert(np.cumsum(n_points), 0, 0)[:-1]
    
    cells = []
    for length, offset in zip(n_points, offsets):
        cells.append(np.hstack([[length], np.arange(offset, offset + length)]))
    lines = np.hstack(cells)
    
    # calculate tangents
    tangents = []
    for s in streamlines:
        if len(s) < 2: 
            tangents.append(np.array([[0,0,0]]))
            continue
        vecs = np.diff(s, axis=0)
        vecs = np.vstack([vecs, vecs[-1:]])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tangents.append(vecs / norms)
        
    return points, lines, np.vstack(tangents)
