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

yab.plot_cortical()
yab.plot_subcortical()
yab.plot_tracts()