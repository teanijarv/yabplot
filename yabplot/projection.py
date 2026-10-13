import warnings
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

def project_vol2surf(nii_path, bmesh='midthickness', mask_medial_wall=True, interpolation='linear'):
    """
    Projects a 3D NIfTI volume onto 2D cortical surface vertices.

    It maps volumetric data directly onto surface meshes by converting real-world coordinates
    using the image affine and sampling the data array at those exact points.

    Parameters
    ----------
    nii_path : str
        absolute path to the 3D or 4D NIfTI volume.
        if 4D, only the first volume/timepoint is used.
    bmesh : str, dict, or pyvista.PolyData, optional
        background mesh to use for projection coordinates. accepts a standard
        string (e.g., 'midthickness') or a dictionary of custom pyvista meshes
        {'L': mesh, 'R': mesh}. default is 'midthickness'.
    mask_medial_wall : bool, optional
        whether to automatically set the medial wall vertices to NaN to prevent
        subcortical signal from bleeding onto the cortical surface.
        Note: only supported if `bmesh` is a standard string. default is True.
    interpolation : {'linear', 'nearest'}, optional
        interpolation method for sampling the volume. 'linear' performs trilinear
        interpolation (smoother, good for continuous t-stats), while 'nearest'
        snaps to the closest voxel center (strictly required for p-values or atlases).
        default is 'linear'.

    Returns
    -------
    lh_data : numpy.ndarray
        1D array of projected values for the left hemisphere vertices.
    rh_data : numpy.ndarray
        1D array of projected values for the right hemisphere vertices.

    Notes
    -----
    The volume is sampled at the vertex coordinates of the mesh using the image
    affine, without any registration, so the volume and the mesh must be in the
    same space. The standard meshes are in MNI152NLin6Asym space (see
    `yabplot.mesh.load_bmesh`). For a volume in another space, pass meshes in
    that space as `bmesh`.
    """
    from .data import get_surface_paths
    from .mesh import load_bmesh, extract_polydata

    # load volume
    img = nib.load(nii_path)
    vol_data = img.get_fdata()

    if vol_data.ndim > 3:
        warnings.warn(f"[WARNING] detected {vol_data.ndim}d nifti volume. using the first volume (index 0).")
        vol_data = vol_data[..., 0]

    inv_affine = np.linalg.inv(img.affine)

    # load brain mesh
    loaded_meshes = load_bmesh(bmesh)

    if interpolation not in ['linear', 'nearest']:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    if 'L' not in loaded_meshes or 'R' not in loaded_meshes:
        raise ValueError("project_vol2surf requires both 'L' and 'R' hemispheres in the bmesh dictionary.")

    # extract raw coordinates for math
    lh_v, _ = extract_polydata(loaded_meshes['L'])
    rh_v, _ = extract_polydata(loaded_meshes['R'])

    def sample_surface(vertices, volume, inv_aff, interp):
        # convert [x, y, z] to [x, y, z, 1] to allow 4x4 affine matrix multiplication
        coords_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        # multiply by inverse affine to get exact decimal voxel coordinates
        vox_coords = inv_aff.dot(coords_homo.T)[:3, :]
        # set scipy interpolation order (1 = trilinear, 0 = nearest neighbor)
        order = 1 if interp == 'linear' else 0
        # sample the 3d volume at the calculated decimal coordinates
        return map_coordinates(volume, vox_coords, order=order, mode='nearest')

    # projection
    lh_data = sample_surface(lh_v, vol_data, inv_affine, interpolation)
    rh_data = sample_surface(rh_v, vol_data, inv_affine, interpolation)

    # handle the medial wall
    if mask_medial_wall:
        if isinstance(bmesh, str):
            # only if standard fs_LR 32k mesh
            lh_mask_path, rh_mask_path = get_surface_paths('nomedialwall', 'label')
            lh_data[nib.load(lh_mask_path).darrays[0].data == 0] = np.nan
            rh_data[nib.load(rh_mask_path).darrays[0].data == 0] = np.nan
        else:
            warnings.warn("[WARNING] medial wall masking is only automatically supported for standard yabplot string meshes. skipping mask.")

    return lh_data, rh_data


def project_vol2tract_atlas(nii_path, atlas='xtract_tiny', custom_atlas_path=None, interpolation='linear'):
    """
    Samples a 3D volume across all tracts in a specific atlas.
    This is a convenience function around `project_vol2tract` that automatically
    resolves the atlas paths, loops through all available tractograms, and returns
    a dictionary ready to be passed directly to `plot_tracts`.

    Parameters
    ----------
    nii_path : str
        absolute path to the 3D nifti volume (e.g., FA or MD map).
    atlas : str, optional
        name of the standard tract atlas. default is 'xtract_tiny'.
    custom_atlas_path : str, optional
        path to a custom directory of .trk files.
    interpolation : {'linear', 'nearest'}, optional
        trilinear interpolation (default) blends nearby voxels for a smooth map.

    Returns
    -------
    dict
        dictionary mapping tract names to their 1D sampled data arrays.
    """
    from .data import _resolve_resource_path, _find_tract_files

    # resolve the atlas directory and locate all tract files
    atlas_dir = _resolve_resource_path(atlas, 'tracts', custom_path=custom_atlas_path)
    tract_files = _find_tract_files(atlas_dir)

    tract_data = {}

    # loop through and map the volume to each tract
    for name, trk_path in tract_files.items():
        tract_data[name] = project_vol2tract(trk_path, nii_path, interpolation)

    return tract_data


def project_vol2tract(trk_path, nii_path, interpolation='linear'):
    """
    Samples a 3D volume natively at every vertex of a tractogram. Maps the streamline
    coordinates directly into the volumetric voxel space using the image affine.

    Parameters
    ----------
    trk_path : str
        absolute path to the .trk or .tck tractography file.
    nii_path : str
        absolute path to the 3D nifti volume (e.g., FA or MD map).
    interpolation : {'linear', 'nearest'}, optional
        trilinear interpolation (default) blends nearby voxels for a smooth map.

    Returns
    -------
    numpy.ndarray
        1D array of sampled values corresponding exactly to the flattened
        points of the tractogram, ready to be injected into plot_tracts.
    """
    if interpolation not in ['linear', 'nearest']:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    # load the 3D volume
    img = nib.load(nii_path)
    vol_data = img.get_fdata()
    if vol_data.ndim > 3:
        vol_data = vol_data[..., 0]

    inv_affine = np.linalg.inv(img.affine)

    # load the tractogram
    trk = nib.streamlines.load(trk_path)

    # stack all streamline coordinates into a single (n_points, 3) array
    points = np.vstack(trk.streamlines)

    # convert coordinates using the inverse affine
    coords_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    vox_coords = inv_affine.dot(coords_homo.T)[:3, :]

    # sample the volume
    order = 1 if interpolation == 'linear' else 0
    sampled_data = map_coordinates(vol_data, vox_coords, order=order, mode='nearest')

    return sampled_data
