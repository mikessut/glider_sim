import numpy as np
import matplotlib.pyplot as plt


G = 9.81  # m / s^2
RHO_AIR = 1.2  # sea level [=] kg / m^3


def rotM(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


class GliderPitchSim:

    _mass = 23e-3  # kg
    _momI = 20e-2**2 * 10e-3  # guess: (20cm)^2 * 10g  (kg m ^2)

    # Angles all in radians
    _wing_alpha = 4 * np.pi / 180
    _horz_stab_alpha = -4 * np.pi / 180

    # pos center of lift (behind CG)
    _pos_wing = 20e-3
    _pos_horz_stab = 40e-2

    # lifting surface areas
    _wing_A = 60e-2 * 7e-2
    _horz_stab_A = 14e-2 * 7e-2

    _dt = 0.005

    def __init__(self) -> None:
        self._t = []
        self._r = []  # x, y position  m
        self._v = []  # x, y velocity m/s^2
        self._pitch = []  # rad
        self._w = [] # pitch velocity

        self._Fwing = []
        self._Fstab = []

        self._mom = []
        self._ld = []
        self._lift = []
        self._drag = []

        self._a = []


    def init_conditions(self, velocity, pitch, w=0):
        self._t.append(0.0)
        self._r.append(np.array((0, 0), dtype=float))
        self._v.append(np.array(velocity, dtype=float))
        self._a.append(np.array((0, 0), dtype=float))
        self._pitch.append(pitch)
        self._w.append(w)

    def finalize_lists(self):
        self._r = np.array(self._r)
        self._v = np.array(self._v)
        self._a = np.array(self._a)
        self._pitch = np.array(self._pitch)
        self._w = np.array(self._w)
        self._t = np.array(self._t)
        self._Fwing = np.array(self._Fwing)
        self._Fstab = np.array(self._Fstab)
        self._mom = np.array(self._mom)
        self._ld = np.array(self._ld)
        self._lift = np.array(self._lift)
        self._drag = np.array(self._drag)

    def step(self):
        pitch = self._pitch[-1]
        aoa = pitch - np.arctan2(self._v[-1][1], self._v[-1][0])
        vmag = np.linalg.norm(self._v[-1])

        # Calculate lift and drag components.  These act perpendicular and
        # parallel to the free streamline respectively.
        Flift_wing = self.Clift(aoa + self._wing_alpha) * self._wing_A * .5 * RHO_AIR * vmag**2
        Fdrag_wing = self.Cdrag(aoa + self._wing_alpha) * self._wing_A * .5 * RHO_AIR * vmag**2

        # Rotate into body frame (rotate by AoA) (longitudinal axis, vert axis)
        Fwing = rotM(-aoa - self._wing_alpha) @ np.vstack((-Fdrag_wing, Flift_wing))

        # Thrust can be added here
        # Fwing[0] += self._mass * 5

        # Repeat for horizontal stabilizer
        Flift_horz_stab = self.Clift(aoa + self._horz_stab_alpha) * self._horz_stab_A * .5 * RHO_AIR * vmag**2
        Fdrag_horz_stab = self.Cdrag(aoa + self._horz_stab_alpha) * self._horz_stab_A * .5 * RHO_AIR * vmag**2

        Fstab = rotM(-aoa - self._horz_stab_alpha) @ np.vstack((-Fdrag_horz_stab, Flift_horz_stab))

        self._Fwing.append(Flift_wing)
        self._Fstab.append(Flift_horz_stab)

        # Rotate forces into inertial frame
        Fwing_int = rotM(pitch) @ Fwing
        Fstab_int = rotM(pitch) @ Fstab
        self._lift.append(Fwing_int[1, 0] + Fstab_int[1, 0])
        self._drag.append(Fwing_int[0, 0] + Fstab_int[0, 0])
        self._ld.append((Fwing_int[1, 0] + Fstab_int[1, 0]) / (Fwing_int[0, 0] + Fstab_int[0, 0]))

        # Sum forces and moments to solve for linear and rotational accelerations
        a = ((Fwing_int + Fstab_int - np.vstack([0, self._mass * G])) / self._mass).flatten()
        self._a.append(a)
        #print(a)
        # import pdb; pdb.set_trace()
            
        # Sum moments about CG and solve for rot acc [=] 1/s^2
        # Positive rotation is pitch up
        w_acc = -(Fwing[1, 0] * self._pos_wing + Fstab[1, 0] * self._pos_horz_stab) / self._momI
        self._mom.append((-Fwing[1, 0] * self._pos_wing, -Fstab[1, 0] * self._pos_horz_stab))
        # w_acc *= 0

        # print("moments: wing, stab", Flift_wing * self._pos_center_lift, Flift_horz_stab * self._pos_horz_stab)
        # print("ax, ay, w_acc", ax, ay, w_acc)

        # import pdb; pdb.set_trace()
        self._v.append(self._v[-1] + a * self._dt)
        self._r.append(self._r[-1] + self._v[-1] * self._dt + 0.5 * a * self._dt**2)

        self._w.append(self._w[-1] + w_acc * self._dt)
        self._pitch.append(self._pitch[-1] + self._w[-1] * self._dt + 0.5 * w_acc * self._dt**2)

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
        
        Cd = 1.28 * np.sin(np.abs(alpha))
        
        if scalar_input:
            return np.squeeze(Cd)
        return Cd

    def plot(self):
        
        f, ax = plt.subplots(4, 1, sharex=True, figsize=(7, 10))
        #ax[0].plot(self._t, self._r[:, 0])
        ax[0].plot(self._t, self._r[:, 1])
        ax[0].set_ylabel('y')

        ax[1].plot(self._t, self._v[:, 0])
        ax[1].plot(self._t, self._v[:, 1])
        ax[1].plot(self._t, np.linalg.norm(self._v, axis=1))
        ax[1].set_ylabel('x, y vel')

        ax[2].plot(self._t, self._pitch*180/np.pi)
        ax[2].plot(self._t, (self._pitch - np.arctan2(self._v[:, 1], self._v[:, 0])) * 180 / np.pi)
        ax[2].grid(True)
        ax[2].set_ylabel('Pitch, AoA (deg)')

        # ax[3].plot(self._t[1:], self._Fwing)
        # ax[3].plot(self._t[1:], self._Fstab)
        # ax[3].axhline(self._mass * G)
        #ax[3].plot(self._t, self._w * 180/np.pi)
        #ax[3].set_ylabel(r'$\omega$ (dps)')

        # ax[4].plot(self._t[1:], self._mom[:, 1])  # Horz Stab
        ax[3].plot(self._t[1:], self._mom.sum(axis=1), label='sum')
        ax[3].plot(self._t[1:], self._mom[:, 0], label='wing')
        ax[3].plot(self._t[1:], self._mom[:, 1], label='stab')
        ax[3].set_ylabel('Mom')
        ax[3].grid(True)
        ax[3].legend(loc=0)
        ax[3].set_xlabel('Time (sec)')
        #ax[4].set_ylim((-.005, .005))
