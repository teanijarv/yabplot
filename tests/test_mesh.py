import pytest
import numpy as np
import pyvista as pv
from yabplot.mesh import load_nii_as_mesh, make_cortical_mesh, get_region_boundaries

def test_load_nii_as_mesh(synthetic_nifti):
    """Verify that marching cubes mesh extraction works on a NIfTI volume."""
    mesh = load_nii_as_mesh(synthetic_nifti, threshold=5.0, blur_sigma=0, smooth_i=0)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0
    assert mesh.n_cells > 0

def test_make_cortical_mesh():
    """Verify that a manual mesh can be constructed from vertex and face arrays."""
    verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
    faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int64)
    scalars = np.array([1, 2, 3, 4], dtype=np.float32)
    mesh = make_cortical_mesh(verts, faces, scalars, scalar_name='TestData')

    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 4
    assert mesh.n_cells == 4
    assert 'TestData' in mesh.point_data
    assert np.all(mesh.point_data['TestData'] == scalars)

def test_get_region_boundaries():
    """Verify that a region border is traced between the two regions, not along mesh edges."""
    plane = pv.Plane(i_size=10, j_size=10, i_resolution=10, j_resolution=10).triangulate()
    verts, faces = plane.points, plane.faces.reshape(-1, 4)[:, 1:]
    # region 1 ends at x=-1 and region 2 starts at x=0
    labels = np.where(verts[:, 0] < -0.5, 1, 2)

    lines = get_region_boundaries(verts, faces, labels)
    assert lines.n_lines > 0
    assert np.all((lines.points[:, 0] > -1) & (lines.points[:, 0] < 0))
    assert get_region_boundaries(verts, faces, labels, region_ids=[3]) is None
