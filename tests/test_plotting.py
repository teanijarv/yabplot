import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
import yabplot as yab
import pyvista as pv
from yabplot.plotting import get_base_name

# tell PyVista to run in "off-screen" mode so it doesn't try to open a real window
pv.OFF_SCREEN = True

def test_version():
    """Check that the package has a version string."""
    assert yab.__version__ is not None

def test_none_returns_empty_dict():
    """Verify that passing None disables the background mesh."""
    result = yab.mesh.load_bmesh(None)
    assert result == {}

def test_dict_passthrough():
    """Verify that custom dictionary keys for hemispheres are properly sanitized."""
    mesh_l = pv.Sphere()
    mesh_r = pv.Cube()
    mesh_other = pv.Cone()
    d = {'left': mesh_l, 'RIGHT': mesh_r, 'other': mesh_other}
    result = yab.mesh.load_bmesh(d)
    expected = {'L': mesh_l, 'R': mesh_r, 'other': mesh_other}
    assert result == expected

def test_polydata_wrapped_in_both():
    """Verify that passing a single PyVista mesh safely wraps it."""
    mesh = pv.Sphere()
    result = yab.mesh.load_bmesh(mesh)
    assert 'both' in result
    assert result['both'] is mesh

def test_plotter_instantiation():
    """Smoke test: can we create a Plotter without crashing?"""
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    plotter.show()
    plotter.close()

def test_plot_cortical():
    """Smoke test: verify that plot_cortical renders with standard atlas."""
    yab.plot_cortical(atlas='aparc', display_type='matplotlib')

def test_plot_subcortical():
    """Smoke test: verify that plot_subcortical renders with standard atlas."""
    yab.plot_subcortical(atlas='aseg', display_type='matplotlib')

def test_plot_tracts():
    """Smoke test: verify that plot_tracts renders with standard atlas."""
    yab.plot_tracts(atlas='xtract_tiny', display_type='matplotlib')

def test_plot_vertexwise():
    """Smoke test: verify that plot_vertexwise renders with custom meshes."""
    lh = pv.Sphere()
    rh = pv.Sphere()
    lh['Data'] = np.random.rand(lh.n_points)
    rh['Data'] = np.random.rand(rh.n_points)
    yab.plot_vertexwise(lh, rh, display_type='matplotlib')

def test_plot_voxelwise(synthetic_nifti):
    """Smoke test: verify that plot_voxelwise renders with synthetic volume."""
    yab.plot_voxelwise(synthetic_nifti, threshold=5.0, blur_sigma=0, display_type='matplotlib')

def test_plot_connectome():
    """Smoke test: verify that plot_connectome renders both nodes-only and matrix modes."""
    yab.plot_connectome(atlas='aparc', display_type='matplotlib')
    # plotting with a dummy matrix
    regions = yab.get_atlas_regions('aparc', 'cortical')
    n = len(regions)
    matrix = np.random.rand(n, n)
    yab.plot_connectome(matrix=matrix, atlas='aparc', edge_threshold=0.5, display_type='matplotlib')


def test_matplotlib_ax_compatibility():
    """Verify that plotting on a provided Matplotlib axis works correctly."""
    fig, ax = plt.subplots()
    ret_ax = yab.plot_cortical(atlas='aparc', ax=ax, display_type='matplotlib')
    assert ret_ax is ax
    # verify ax is usable
    ret_ax.set_title("Test Title")
    assert ret_ax.get_title() == "Test Title"
    plt.close(fig)

def test_export_path(tmp_path):
    """Verify that export_path correctly saves a file to disk."""
    out_file = tmp_path / "test_export.png"
    yab.plot_cortical(atlas='aparc', display_type='matplotlib', export_path=str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 0


# --- get_base_name unit tests ---

@pytest.mark.parametrize("name,expected", [
    # suffix patterns
    ("putamen_l", "putamen"),
    ("putamen_L", "putamen"),
    ("Putamen_r", "Putamen"),
    ("hippocampus-lh", "hippocampus"),
    ("hippocampus-rh", "hippocampus"),
    # prefix patterns
    ("l-thalamus", "thalamus"),
    ("r-thalamus", "thalamus"),
    ("L_Thalamus", "Thalamus"),
    ("R_Thalamus", "Thalamus"),
    # word prefix patterns
    ("left_caudate", "caudate"),
    ("right_caudate", "caudate"),
    ("left-amygdala", "amygdala"),
    ("right-amygdala", "amygdala"),
    # no hemisphere tag — returned as-is
    ("brainstem", "brainstem"),
    ("cerebellum", "cerebellum"),
])
def test_get_base_name(name, expected):
    assert get_base_name(name) == expected


# --- shared_hemisphere_colors behavioral tests ---

def _count_unique_region_colors(shared_hemisphere_colors):
    """Helper: run plot_subcortical and return the count of unique region colors."""
    from unittest.mock import patch

    captured = {}
    original_add_mesh = pv.Plotter.add_mesh

    def recording_add_mesh(self, mesh, **kwargs):
        color = kwargs.get('color')
        if color is not None:
            captured[id(mesh)] = color
        return original_add_mesh(self, mesh, **kwargs)

    with patch.object(pv.Plotter, 'add_mesh', recording_add_mesh):
        yab.plot_subcortical(
            atlas='aseg', display_type='matplotlib', shared_hemisphere_colors=shared_hemisphere_colors
        )

    return len(set(
        c if not isinstance(c, list) else tuple(c)
        for c in captured.values()
    ))


def test_subcortical_shared_hemisphere_colors_false_more_unique_than_true():
    """shared_hemisphere_colors=False (default) yields more unique colors than True."""
    unique_false = _count_unique_region_colors(shared_hemisphere_colors=False)
    unique_true = _count_unique_region_colors(shared_hemisphere_colors=True)
    assert unique_false > unique_true


def test_subcortical_shared_hemisphere_colors_smoke():
    """Smoke test: shared_hemisphere_colors=True renders without error."""
    yab.plot_subcortical(atlas='aseg', display_type='matplotlib', shared_hemisphere_colors=True)


def test_subcortical_atlas_color_seed_smoke():
    """Smoke test: a non-default atlas_color_seed renders without error."""
    yab.plot_subcortical(atlas='aseg', display_type='matplotlib', atlas_color_seed=7)


def _capture_region_colors(atlas_color_seed, cmap):
    """Helper: return the per-region colors add_mesh receives, in call order."""
    from unittest.mock import patch

    captured = []
    original_add_mesh = pv.Plotter.add_mesh

    def recording_add_mesh(self, mesh, **kwargs):
        color = kwargs.get('color')
        if color is not None:
            captured.append(tuple(color) if not isinstance(color, str) else color)
        return original_add_mesh(self, mesh, **kwargs)

    with patch.object(pv.Plotter, 'add_mesh', recording_add_mesh):
        yab.plot_subcortical(atlas='aseg', display_type='matplotlib',
                             cmap=cmap, atlas_color_seed=atlas_color_seed)
    return captured


def test_atlas_color_seed_changes_colormap_arrangement():
    """With a colormap, different seeds produce different region color arrangements."""
    a = _capture_region_colors(42, 'viridis')
    b = _capture_region_colors(7, 'viridis')
    assert a and b, "no region colors were captured"
    assert a != b, "different atlas_color_seed values should reshuffle colormap colors"


def test_subcortical_plot_regions_separately(tmp_path):
    """plot_regions_separately=True writes one PNG per region+view combination."""
    yab.plot_subcortical(
        atlas='aseg',
        display_type='matplotlib',
        plot_regions_separately=True,
        export_path=str(tmp_path),
    )
    png_files = list(tmp_path.glob("*.png"))
    assert len(png_files) > 0, "No PNG files were written"
    for f in png_files:
        assert f.stat().st_size > 0, f"{f.name} is empty"


def test_subcortical_cmap_object_atlas_mode():
    """A Colormap object (not just a str) must work in atlas mode (data=None).

    The docstring advertises cmap as 'str or matplotlib.colors.Colormap'; passing
    an object previously left color_map undefined and raised NameError.
    """
    yab.plot_subcortical(atlas='aseg', display_type='matplotlib', cmap=plt.get_cmap('viridis'))


def test_plot_regions_separately_midline_no_collision(tmp_path):
    """Midline regions must not overwrite their own left/right views.

    view_face strips 'left_'/'right_', so for a midline region (rendered in every
    view) left_lateral and right_lateral would collapse to the same filename. The
    fix keeps the full view name for regions without a hemisphere tag.
    """
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    # one midline region (no L/R token) and one lateralized region
    pv.Sphere(center=(0, 0, 0)).save(str(atlas_dir / "brainstem.vtk"))
    pv.Sphere(center=(-20, 0, 0)).save(str(atlas_dir / "Left-Putamen.vtk"))

    out = tmp_path / "out"
    yab.plot_subcortical(
        custom_atlas_path=str(atlas_dir),
        display_type='matplotlib',
        plot_regions_separately=True,
        export_path=str(out),
        views=['left_lateral', 'right_lateral'],
    )

    midline_files = sorted(f.name for f in out.glob("brainstem_*.png"))
    # rendered in both lateral views -> must be two distinct files, not one overwritten
    assert len(midline_files) == 2, f"midline region collided: {midline_files}"


def test_plot_regions_separately_displays_without_export(tmp_path, monkeypatch):
    """Without export_path, per-region mode displays and returns axes, writing no files."""
    import matplotlib.axes
    monkeypatch.chdir(tmp_path) 

    result = yab.plot_subcortical(
        atlas='aseg', display_type='matplotlib',
        plot_regions_separately=True, views=['superior'],
    )

    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(a, matplotlib.axes.Axes) for a in result)
    assert list(tmp_path.glob("*.png")) == [], "no files should be written without export_path"
    plt.close('all')
