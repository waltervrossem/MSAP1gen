import numpy as np
import itertools as itt
'''
A simple spot model.

Copyright (c) 2023 Cilia Damiani

This is a free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this code.  If not, see <http://www.gnu.org/licenses/>.

'''

class OneSpot:
    def __init__(self, t, lnprot, Domega=None, rsp=None, latsp=None, lonsp=None, t0=None, lifetime=None, fs=None):
        """
        Creation of spot with attributes radius, latitude and longitude in time
        :param t: the vector time of the light curve
        :type t: np.array
        :param lnprot: the logarithm of the period in days
        :type lnprot: float
        :param Domega: the relative differential rotation
        :type Domega: float
        :param rsp: the angular diameter of the spot in deg
        :type rsp: float
        :param latsp: the latitude of the spot in deg
        :type latsp: float
        :param lonsp: the longitude diameter of the spot in deg
        :type lonsp: float
        :param t0: the time of maximum spot contrast
        :type t0: float
        :param lifetime: the lifetime of the spot in unit of prot
        :type lifetime: float
        :param fs: the maximum constrast of the spot
        :type fs: float
        """
        if Domega is None:
            Domega = 0.0
        if rsp is None:
            rsp = 2.5
        if latsp is None:
            latsp = 0.0
        if lonsp is None:
            lonsp = 0.0
        if t0 is None:
            t0 = 0.0
        if lifetime is None:
            lifetime = np.inf
        if fs is None:
            fs = 0.7

        self.alpha = np.radians(rsp)
        self.chi = np.radians(latsp)
        self.psi0 = np.radians(lonsp)

        self.t0 = t0
        self.lt = lifetime

        Omspot = 2.0 * np.pi * (1.0 - Domega * np.sin(self.chi) * np.sin(self.chi)) / np.exp(lnprot)

        self.psi = self.psi0 + (t - self.t0) * Omspot
        self.duree = np.exp(-(np.log(2.0) * (t - self.t0) / (np.exp(lnprot)*lifetime)) ** 2.0)  # Mosser 2009

        self.fs = fs
        self.t = t

    def compab(self, incl=None, modul=None):
        if incl is None:
            incl = 90.0

        # Compute factors A and B defined by Eq.5 in Dorren 1987
        alpha_max = self.alpha
        chi = self.chi
        psi = self.psi
        N = self.psi.size
        i = np.radians(incl)  # inclination in radians
        AB = np.zeros((N, 2))
        # Following Eq.8 in Dorren 1987
        beta = np.arccos(np.cos(i) * np.sin(chi) + np.sin(i) * np.cos(chi) * np.cos(psi))
        # Test for the visibility of the spot
        for idx, bet in enumerate(beta, start=0):
            if modul > 0.0:
                alpha = alpha_max * (1.0 + 0.3 * np.sin(2.0 * np.pi * self.t[idx] / modul))
            else:
                alpha = alpha_max
            if bet + alpha <= np.pi / 2.0:
                # the spot is completely in view
                delta = 0.0
                zeta = 0.0
            else:
                # the spot is partially in view
                # following Eq7 in Dorren 1987
                delta = np.arccos(1. / (np.tan(alpha) * np.tan(bet)))
                zeta = np.arctan2(np.sin(delta) * np.sin(alpha), np.cos(alpha) / np.sin(bet))

            if bet <= np.pi / 2.0:
                T = np.arctan(np.sin(zeta) * np.tan(bet))
            else:
                T = np.pi - np.arctan(-np.sin(zeta) * (np.tan(bet)))

            if bet - alpha >= np.pi / 2.0:
                # the spot is completely out of view
                AB[idx, :] = np.zeros(2)
            else:
                AB[idx, 0] = zeta + (np.pi - delta) * np.cos(bet) * (np.sin(alpha) ** 2.0) \
                             - (np.sin(zeta) * np.sin(bet) * np.cos(alpha))
                AB[idx, 1] = (-1.0 / 3.0) * (np.pi - delta) * \
                             (2.0 * pow(np.cos(alpha), 3)
                              + (3.0 * pow(np.sin(bet), 2) * np.cos(alpha) * pow(np.sin(alpha), 2))) \
                             + (2.0 / 3.0) * (np.pi - T) \
                             + (1.0 / 6.0) * (np.sin(zeta) * np.sin(2.0 * bet) * (2.0 - 3.0 * pow(np.cos(alpha), 2)))

        return AB

    def dimming(self, incl=None, mue=None, mus=None, fe=None, modul=None):
        # Computes the integrated flux accounting for the visible spot area

        if mue is None:
            mue = 0.59
        if mus is None:
            mus = 0.78
        if fe is None:
            fe = 1.0

        AB = self.compab(incl, modul)
        aa = (1.0 - mue) - ((1.0 - mus) * self.fs / fe)
        bb = mue - (mus * self.fs / fe)

        return (aa * AB[:, 0] + bb * AB[:, 1]) / (np.pi * (1.0 - mue / 3.0)) * self.duree


def dimlist(listspot, incl, mue, mus, modul):
    nspt = len(listspot)
    nt = len(listspot[0].psi)
    dimmings = np.empty((nspt, nt))
    for i in range(0, nspt):

        dimmings[i, :] = listspot[i].dimming(incl=incl, mue=mue, mus=mus, modul=modul)
    return dimmings

def testoverlap(tach1, tach2):
    a1 = tach1.alpha
    a2 = tach2.alpha
    p1 = tach1.psi
    p2 = tach2.psi
    c1 = tach1.chi
    c2 = tach2.chi

    S = np.arccos(np.sin(c1) * np.sin(c2) + np.cos(c1) * np.cos(c2) * np.cos(p2 - p1))

    return S < a1 + a2

def paramtolc(defoo, t, nspots, verbose=False):

    inispots = [OneSpot(t, defoo[1], Domega=defoo[3], rsp=defoo[4], latsp=defoo[4 + nspots],
                    lonsp=defoo[4 + 2 * nspots], t0=defoo[4 + 3 * nspots], lifetime=defoo[4 + 4 * nspots], fs=defoo[4 + 5 * nspots])]
    if nspots > 1:
        for i in range(1, nspots):
            ispot = OneSpot(t, defoo[1], Domega=defoo[3], rsp=defoo[4 + i], latsp=defoo[4 + nspots + i],
                       lonsp=defoo[4 + 2 * nspots + i], t0=defoo[4 + 3 * nspots + i],
                       lifetime=defoo[4 + 4 * nspots + i], fs=defoo[4 + 5 * nspots + i])
            inispots.append(ispot)

    for combi in itt.combinations(inispots, 2):
        if np.any(testoverlap(combi[0], combi[1])):
            print("Overlapping spots, aborting")
            ovl = 1
            return 0, inispots, ovl
    else:
        if(verbose):
            print("No Overlapping spots") # executed if the loop ended normally (no break)
        ovl = 0
        flx = 1.0 - np.sum(dimlist(inispots, incl=defoo[2], mue=defoo[5 + 6 * nspots], mus=defoo[6 + 6 * nspots], modul=defoo[7 + 6 * nspots]), axis=0) - defoo[4 + 6 * nspots]
        return flx, inispots, ovl
