import os
import pytest
import numpy as np
import nibabel as nib
import pyvista as pv
from yabplot.atlas_builder import build_subcortical_atlas

pv.OFF_SCREEN = True


def _make_atlas_nifti(tmp_path, label_id, label_value, dtype=np.float64):
    """Create a synthetic atlas NIfTI with a sphere of voxels set to label_value."""
    shape = (20, 20, 20)
    data = np.zeros(shape, dtype=dtype)
    cz, cy, cx = 10, 10, 10
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    region_mask = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 5 ** 2
    data[region_mask] = label_value
    path = str(tmp_path / f"atlas_{label_id}.nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def test_build_subcortical_atlas_creates_vtk_and_lut(tmp_path):
    """Verify that VTK mesh and atlas_LUT.txt are created for a matching region."""
    nii_path = _make_atlas_nifti(tmp_path, label_id=10, label_value=10.0)
    out_dir = str(tmp_path / "out")

    build_subcortical_atlas(nii_path, {10: "thalamus_l"}, out_dir, smooth_i=0)

    assert os.path.isfile(os.path.join(out_dir, "thalamus_l.vtk"))
    assert os.path.isfile(os.path.join(out_dir, "atlas_LUT.txt"))


def test_build_subcortical_atlas_rounds_float_labels(tmp_path):
    """Check that float-imprecise label values are correctly matched after .round().

    NIfTI volumes often store integer labels as float32, which can lead to slight 
    imprecision when read back as float64 (using, e.g., nibabel.load). For example, 
    a label value of 10.0 stored as float32 may be read back as 10.000000953674316 in float64.
    Without .round(), the mask `data == rid` misses these voxels and produces no mesh. 
    This test confirms the rounding fix is in effect.
    
    """
    # float32(10) + float32(1e-6) → 10.000001 in float32 → 10.000000953674316 in float64
    label_value = np.float32(10.0) + np.float32(1e-6)
    nii_path = _make_atlas_nifti(tmp_path, label_id=10, label_value=label_value, dtype=np.float32)

    # Confirm the fixture actually has float-imprecise values (not exactly 10)
    loaded_vals = np.unique(nib.load(nii_path).get_fdata())
    loaded_vals = loaded_vals[loaded_vals != 0]
    assert not np.all(loaded_vals == 10), (
        "fixture should have float-imprecise label values to test rounding"
    )

    out_dir = str(tmp_path / "out")
    build_subcortical_atlas(nii_path, {10: "thalamus_l"}, out_dir, smooth_i=0)

    assert os.path.isfile(os.path.join(out_dir, "thalamus_l.vtk")), (
        "thalamus_l.vtk not created — .round() may be missing from the get_fdata() call"
    )


def test_build_subcortical_atlas_missing_region_skips(tmp_path):
    """Verify that a label absent from the volume produces no VTK file."""
    nii_path = _make_atlas_nifti(tmp_path, label_id=10, label_value=10.0)
    out_dir = str(tmp_path / "out")

    build_subcortical_atlas(nii_path, {99: "phantom"}, out_dir, smooth_i=0)

    assert not os.path.isfile(os.path.join(out_dir, "phantom.vtk"))


def test_build_subcortical_atlas_both_filters_raises(tmp_path):
    """Verify that providing both include_list and exclude_list raises ValueError."""
    nii_path = _make_atlas_nifti(tmp_path, label_id=10, label_value=10.0)

    with pytest.raises(ValueError):
        build_subcortical_atlas(
            nii_path,
            {10: "thalamus_l"},
            str(tmp_path / "out"),
            include_list=["thalamus"],
            exclude_list=["cerebellum"],
        )
