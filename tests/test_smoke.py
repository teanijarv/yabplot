import pytest
import yabplot as yab
import pyvista as pv

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
    yab.plot_cortical(atlas='aparc', display_type='none')

def test_plot_subcortical():
    """
    Integration test: Downloads 'aseg' and plots it.
    """
    yab.plot_subcortical(atlas='aseg', display_type='none')

def test_plot_tracts():
    """
    Integration test: Downloads 'xtract_tiny' and plots it.
    """
    yab.plot_tracts(atlas='xtract_tiny', display_type='none')