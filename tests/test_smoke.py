import yabplot as yab
import pyvista as pv
import numpy as np
import nibabel as nib

# tell PyVista to run in "off-screen" mode so it doesn't try to open a real window
pv.OFF_SCREEN = True

def test_version():
    """Check that the package has a version string."""
    assert yab.__version__ is not None

def test_plotter_instantiation():
    """
    Smoke test: Can we create a Plotter without crashing?
    This verifies VTK and PyVista are correctly linked to the system display.
    """
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show()
    plotter.close()

def test_plot_cortical():
    """
    Integration test: Downloads 'aparc' and plots it.
    """
    yab.plot_cortical(atlas='aparc', display_type=None)

def test_plot_subcortical():
    """
    Integration test: Downloads 'aseg' and plots it.
    """
    yab.plot_subcortical(atlas='aseg', display_type=None)

def test_plot_tracts():
    """
    Integration test: Downloads 'xtract_tiny' and plots it.
    """
    yab.plot_tracts(atlas='xtract_tiny', display_type=None)

def test_plot_voxelwise():
    """
    Integration test: Plot synthetic voxelwise data with numpy array + affine.
    """
    # Create small synthetic voxel data (10x10x10)
    voxel_data = np.random.rand(10, 10, 10).astype(np.float32)
    # Identity affine (voxel space = world space)
    affine = np.eye(4, dtype=np.float32)
    
    # This should not crash
    yab.plot_voxelwise(voxel_data, affine=affine, display_type=None)

def test_plot_voxelwise_4d_nifti():
    """
    Integration test: Plot 4D NIfTI with singleton dimension (should be squeezed to 3D).
    """
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, 'test_4d.nii.gz')
        data_4d = np.random.rand(15, 15, 15, 1).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data_4d, affine)
        nib.save(img, nifti_path)
        
        # Should handle 4D with singleton dimension gracefully
        yab.plot_voxelwise(nifti_path, display_type=None)

def test_plot_voxelwise_mask_background():
    """
    Integration test: Plot binary mask with mask_background parameter.
    """
    binary_mask = (np.random.rand(10, 10, 10) > 0.5).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    
    # Test with mask_background=True (default, should hide 0s)
    yab.plot_voxelwise(binary_mask, affine=affine, mask_background=True, display_type=None)
    
    # Test with mask_background=False (should show all values)
    yab.plot_voxelwise(binary_mask, affine=affine, mask_background=False, display_type=None)