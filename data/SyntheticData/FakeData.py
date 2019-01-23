# based on https://github.com/dhale/ipf/blob/master/src/ipf/FakeData.java

#
import math

import scipy
from scipy.ndimage import affine_transform

import numpy as np
import matplotlib.pyplot as plt
#
#
# def makeReflectivityWithNormals(n1, n2):
#     r = np.random.uniform(-1,1,n1) ** 5.0
#     p = np.zeros(3, n2, n1)
#     for i2 in range(n2):
#         p[0][i2] = r
#     p[1] = np.ones(n1, n2)
#     p[2] = np.zeros(n1, n2)
#     return p
#
# def seismicAndSlopes2d2014A(noise):
#     n1 = 501
#     n2 = 501
#     p = makeReflectivityWithNormals(n1, n2)
#     q = makeReflectivityWithNormals(n1, n2)
#     r = makeReflectivityWithNormals(n1, n2)

a1 = 0.0
b1 = 0.05
a2 = 1.0e-4
b2 = 2.0e-4

x = np.arange(1, 500, 1)
y = np.arange(1, 500, 1)
xx, yy = np.meshgrid(x, y, sparse=True)
#z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
z = xx - (a1 + b1 * xx)*np.sin((a2+b2*yy)*yy)
h = plt.contourf(x,y,z)
plt.show()

transf = affine_transform(z, np.array([[1, 2],[2, 1]]))
h = plt.contourf(x,y,transf)
plt.show()

fx1=0
fx2 = 0.2
tftheta = -15.0
rtheta = np.radians(tftheta)
ttheta = np.tan(rtheta)
ctheta = np.cos(rtheta)
stheta = np.sin(rtheta)
u1 = -stheta
u2 =  ctheta
if u1>0.0:
    u1 = - u1
    u2 = - u2

u0 = -(fx1*u1+fx2*u2)

#_t1 = fthrow

def faulted(x1, x2):
      return u0+u1*x1+u2*x2>=0.0

# for i2 in range(500):
#     for i1 in range(500):
#     C2 f = t.f(i1, i2);
# float f1 = f.c1;
# float f2 = f.c2;
# q[i2][i1] = _si.interpolate(n1, 1.0, 0.0, n2, 1.0, 0.0, p, f1, f2);
# }
# }
