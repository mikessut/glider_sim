import numpy as np
import matplotlib.pyplot as plt


G = 9.81  # m / s^2
RHO_AIR = 1.2  # sea level [=] kg / m^3


def _rotM(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


class GliderPitchSim:

    _mass = 30e-3  # kg
    _momI = 20e-2**2 * 10e-3  # guess: (20cm)^2 * 10g  (kg m ^2)

    # Angles all in radians
    _wing_alpha = 4 * np.pi / 180
    _horz_stab_alpha = -4 * np.pi / 180

    # pos center of lift (behind CG)
    _pos_center_lift = 20e-3
    _pos_horz_stab = 40e-2

    # lifting surface areas
    _wing_A = 60e-2 * 7e-2
    _horz_stab_A = 12e-2 * 7e-2

    _dt = 0.01

    def __init__(self) -> None:
        self._t = []
        self._r = []  # x, y position  m
        self._v = []  # x, y velocity m/s^2
        self._pitch = []  # rad
        self._w = [] # pitch velocity

        self._Fwing = []
        self._Fstab = []

        self._mom = []


    def init_conditions(self, velocity, pitch, w=0):
        self._t.append(0.0)
        self._r.append(np.array((0, 0), dtype=float))
        self._v.append(np.array(velocity, dtype=float))
        self._pitch.append(pitch)
        self._w.append(w)

    def finalize_lists(self):
        self._r = np.array(self._r)
        self._v = np.array(self._v)
        self._pitch = np.array(self._pitch)
        self._w = np.array(self._w)
        self._t = np.array(self._t)
        self._Fwing = np.array(self._Fwing)
        self._Fstab = np.array(self._Fstab)
        self._mom = np.array(self._mom)

    def step(self):
        pitch = self._pitch[-1]
        aoa = pitch - np.arctan(self._v[-1][1] / self._v[-1][0])
        # w = self._w[-1]
        vmag = np.linalg.norm(self._v[-1])
        #vmag = self._v[-1][0]

        Flift_wing = self.Clift(aoa + self._wing_alpha) * self._wing_A * .5 * RHO_AIR * vmag**2
        Fdrag_wing = self.Cdrag(aoa + self._wing_alpha) * self._wing_A * .5 * RHO_AIR * vmag**2

        # rotate into body frame (rotate by AoA) (longitudinal axis, vert axis)
        Fwing = _rotM(-aoa) @ np.vstack((-Fdrag_wing, Flift_wing))
        # print(aoa*180/np.pi, Fwing.flatten())

        Flift_horz_stab = self.Clift(aoa + self._horz_stab_alpha) * self._horz_stab_A * .5 * RHO_AIR * vmag**2
        Fdrag_horz_stab = self.Cdrag(aoa + self._horz_stab_alpha) * self._horz_stab_A * .5 * RHO_AIR * vmag**2

        Fstab = _rotM(-aoa) @ np.vstack((-Fdrag_horz_stab, Flift_horz_stab))

        self._Fwing.append(Flift_wing)
        self._Fstab.append(Flift_horz_stab)

        # Rotate forces into inertial frame
        Fwing_int = _rotM(pitch) @ Fwing
        Fstab_int = _rotM(pitch) @ Fstab
        # print(Fwing_int.flatten())

        a = ((Fwing_int + Fstab_int - np.vstack([0, self._mass * G])) / self._mass).flatten()
        #print(a)
        # import pdb; pdb.set_trace()
            
        # Sum moments about CG and solve for rot acc [=] 1/s^2
        # Making a small pitch angle assumption here
        # Positive rotation is pitch down
        w_acc = (Fwing[1, 0] * self._pos_center_lift + Fstab[1, 0] * self._pos_horz_stab) / self._momI
        self._mom.append((Fwing[1, 0] * self._pos_center_lift, Fstab[1, 0] * self._pos_horz_stab))
        # w_acc *= 0

        # print("moments: wing, stab", Flift_wing * self._pos_center_lift, Flift_horz_stab * self._pos_horz_stab)
        # print("ax, ay, w_acc", ax, ay, w_acc)

        # import pdb; pdb.set_trace()
        self._v.append(self._v[-1] + a * self._dt)
        self._r.append(self._r[-1] + self._v[-1] * self._dt + 0.5 * a * self._dt**2)

        self._w.append(self._w[-1] - w_acc * self._dt)
        self._pitch.append(self._pitch[-1] + self._w[-1] * self._dt - 0.5 * w_acc * self._dt**2)

        self._t.append(self._t[-1] + self._dt)

    def Clift(self, alpha):
        """
        https://www.grc.nasa.gov/www/k-12/airplane/kiteincl.html
        TODO: make more realistic

        Force_lift = Cl * A * .5 * rho * v [=] m^2 * kg / m^3 * m / 2 = kg m / s^2
        """
        alpha = np.asarray(alpha)
        scalar_input = False
        if alpha.ndim == 0:
            alpha = alpha[None]
            scalar_input = True

        Cl = 2 * np.pi * np.sin(alpha)
        alpha_critical = 15*np.pi/180
        deg_flat = 3*np.pi/180
        down_slope = .25 / (5*np.pi/180)

        Cl_critical = 2*np.pi*np.sin(alpha_critical)

        Cl[alpha > alpha_critical] = Cl_critical
        i = alpha > (alpha_critical + deg_flat)
        Cl[i] = Cl_critical - (alpha[i] - (alpha_critical + deg_flat))*down_slope

        if scalar_input:
            return np.squeeze(Cl)
        return Cl

    def Cdrag(self, alpha):
        """
        https://www.grc.nasa.gov/www/k-12/airplane/kiteincl.html
        """
        alpha = np.asarray(alpha)
        scalar_input = False
        if alpha.ndim == 0:
            alpha = alpha[None]
            scalar_input = True
        
        Cd = 1.28 * np.sin(alpha)
        
        if scalar_input:
            return np.squeeze(Cd)
        return Cd
