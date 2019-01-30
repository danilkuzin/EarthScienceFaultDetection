# based on https://github.com/dhale/ipf/blob/master/src/ipf/FakeData.java

import math

import scipy
import scipy.interpolate
from scipy.ndimage import affine_transform

import numpy as np
import matplotlib.pyplot as plt


class T1:
    def f(self, x):
        pass

    def df(self, x):
        pass


class T2:
    def f(self, x1, x2):
        pass

    def df(self, x1, x2):
        pass


class Linear1(T1):
    def __init__(self, a0, a1):
        self.a0 = a0
        self.a1 = a1

    def f(self, x):
        return self.a0 + self.a1 * x

    def df(self, x):
        return self.a1


class C2:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2


class D2:
    def __init__(self, d11, d12, d21, d22):
        self.d11 = d11
        self.d12 = d12
        self.d21 = d21
        self.d22 = d22


class LinearFault2(T2):
    def __init__(self, fx1, fx2, ftheta, fthrow):
        self.r1 = fx1
        self.r2 = fx2

        rtheta = np.radians(ftheta)
        self.ttheta = np.tan(rtheta)

        ctheta = np.cos(rtheta)
        stheta = np.sin(rtheta)
        self.u1 = -stheta
        self.u2 = ctheta

        if self.u1 > 0.0:
            self.u1 = -self.u1
            self.u2 = -self.u2

        self.u0 = -(fx1 * self.u1 + fx2 * self.u2)

        self.t1 = fthrow

    def faulted(self, x1, x2):
        return self.u0 + self.u1 * x1 + self.u2 * x2 >= 0.0

    def f(self, x1, x2):
        if self.faulted(x1, x2):
            x1 -= self.r1
            x2 -= self.r2
            t = self.t1.f(x1)
            x1 -= t
            x2 -= t * self.ttheta
            x1 += self.r1
            x2 += self.r2
        return C2(x1, x2)

    def df(self, x1, x2):
        d11 = 1.0
        d12 = 0.0
        d21 = 0.0
        d22 = 1.0
        if self.faulted(x1, x2):
            x1 -= self.r1
            x2 -= self.r2
            dt = self.t1.df(x1)
            d11 -= dt
            d21 -= dt * self.ttheta
        return D2(d11, d12, d21, d22)


class Sinusoidal2(T2):
    def __init__(self, a1, b1, a2, b2):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

    def f(self, x1, x2):
        f1 = x1 - (self.a1 + self.b1 * x1) * np.sin((self.a2 + self.b2 * x2) * x2)
        f2 = x2
        return C2(f1, f2)

    def df(self, x1, x2):
        d11 = 1.0 - self.b1 * np.sin((self.a2 + self.b2 * x2) * x2)
        d12 = -(self.a1 + self.b1 * x1) * (self.a2 + 2.0 * self.b2 * x2) * np.cos((self.a2 + self.b2 * x2) * x2)
        d21 = 0.0
        d22 = 1.0
        return D2(d11, d12, d21, d22)


class VerticalShear2(T2):
    def __init__(self, s1):
        self.s1 = s1

    def f(self, x1, x2):
        x1 -= self.s1.f(x2)
        return C2(x1, x2)

    def df(self, x1, x2):
        d12 = -self.s1.df(x2)
        return D2(1.0, d12, 0.0, 1.0)


def makeReflectivityWithNormals(n1, n2):
    r = np.random.uniform(-1, 1, n1) ** 5.0
    p = np.zeros((3, n2, n1))
    for i2 in range(n2):
        p[0][i2] = r
    p[1] = np.ones((n1, n2))
    p[2] = np.zeros((n1, n2))
    return p

def apply_im(t, p):
    n2, n1 = p.shape
    q = np.zeros((n2, n1))
    for i2, i1 in zip(range(n2), range(n1)):
        f = t.f(i1, i2)
        f1 = f.c1
        f2 = f.c2
        interp = scipy.interpolate.interp2d(np.linspace(0,n1,n1), np.linspace(0,n2,n2), p)
        q[i2][i1] = interp(f1, f2)
    return q
#
# class SincInterpolator:
#     def __init__(self):
#         self.emax = 0.0
#         self.fmax = 0.3
#         self.lmax = 8


    # def interpolate(self, nx1u, dx1u, fx1u, nx2u, dx2u, fx2u, yu, x1i, x2i):
    #     x1scale = 1.0 / dx1u
    #     x2scale = 1.0 / dx2u
    #     x1shift = _lsinc - fx1u * x1scale
    #     x2shift = _lsinc - fx2u * x2scale
    #     nx1um = nx1u - _lsinc
    #     nx2um = nx2u - _lsinc;
    #     return self.interpolate(
    #         x1scale, x1shift, nx1um, nx1u,
    #         x2scale, x2shift, nx2um, nx2u,
    #         yu, x1i, x2i)
    # }
def apply_im_norm(t, p):
    _, n2, n1 = p.shape
    q0 = apply_im(t,p[0])
    q1 = apply_im(t,p[1])
    q2 = apply_im(t,p[2])
    for i2, i1 in zip(range(n2), range(n1)):
        d = t.df(i1,i2)
        q1i = d.d11*q1[i2][i1]+d.d21*q2[i2][i1]
        q2i = d.d12*q1[i2][i1]+d.d22*q2[i2][i1]
        qsi = 1.0/np.sqrt(q1i*q1i+q2i*q2i)
        q1[i2][i1] = q1i*qsi
        q2[i2][i1] = q2i*qsi
    return np.array((q0,q1,q2))

def seismicAndSlopes2d2014A(noise):
    n1 = 501
    n2 = 501
    p = makeReflectivityWithNormals(n1, n2)
    q = makeReflectivityWithNormals(n1, n2)
    r = makeReflectivityWithNormals(n1, n2)
    throw1 = Linear1(0.0, 0.10)
    throw2 = Linear1(0.0, 0.10)
    fault1 = LinearFault2(0.0, n2 * 0.2, 15.0, throw1)
    fault2 = LinearFault2(0.0, n2 * 0.4, -15.0, throw2)
    fold = Sinusoidal2(0.0, 0.05, 1.0e-4, 2.0e-4)
    shear = VerticalShear2(Linear1(0.0, 0.05))
    p = apply_im_norm(fold, p)
    plt.imshow(p[0])
    plt.show()

if __name__ == "__main__":
    seismicAndSlopes2d2014A(0.5)
