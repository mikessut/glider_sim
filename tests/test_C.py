from glider_model import *
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_Clift():

    alpha = np.linspace(-5, 25, 100) * np.pi/180

    g = GliderPitchSim()
    plt.figure()
    plt.plot(alpha*180/np.pi, g.Clift(alpha))
    plt.xlabel('AoA (deg)')
    plt.ylabel('Cl')
    plt.grid(True)


def test_Cdrag():
    alpha = np.linspace(-5, 25, 100) * np.pi/180

    g = GliderPitchSim()
    plt.figure()
    plt.plot(alpha*180/np.pi, g.Cdrag(alpha))
    plt.xlabel('AoA (deg)')
    plt.ylabel('Cdrag')
    plt.grid(True)


def test_Clift_scalar():
    g = GliderPitchSim()
    Cl = g.Clift(5 * np.pi / 180)
    


@pytest.fixture(scope="session", autouse=True)
def show_plots(request):
    def _show():
        plt.show()
    request.addfinalizer(_show)