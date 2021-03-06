import sympy
from sympy import Lambda, log, Derivative, diff, Rational, simplify, linsolve, lambdify, exp, solve, ask, Q, S
from sympy.abc import x, y
from sympy.plotting import plot3d
 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

f = Lambda((x, y), x**3 - 3*x*y**2 + 18*y)

f_dx = Lambda((x, y), diff(f(x, y), x))
f_2dx = Lambda((x, y), diff(f_dx(x, y), x))

f_dy = Lambda((x, y), diff(f(x, y), y))
f_2dy = Lambda((x, y), diff(f_dy(x, y), y))

f_dxdy = Lambda((x, y), diff(f_dx(x, y), y))

grad_0 = solve([f_dx(x, y), f_dy(x, y)], (x, y), dict=True, domain=S.Reals, real = True)

roots = list()
 
for root in grad_0:
  if ask(Q.real(root[x])) and ask(Q.real(root[x])):
    roots.append((root[x], root[y]))
 
max_min = list()
 
for root in roots:
  A = f_2dx(root[0], root[1])
  B = f_dxdy(root[0], root[1])
  C = f_2dy(root[0], root[1])
 
  expr = A * C - B**2
 
  if True:
    print('({0}, {1}) - '.format(root[0], root[1]), end='')
 
    if expr < 0:
      print('neither minima nor maxima')
      max_min.append((root, f(root[0], root[1]), 0))
    elif expr > 0 and A > 0:
      print('local minima, f =', f(root[0], root[1]))
      max_min.append((root, f(root[0], root[1]), -1))
    elif expr > 0 and A < 0:
      print('local maxima, f =', f(root[0], root[1]))
      max_min.append((root, f(root[0], root[1]), 1))
    else:
      print('unknown')

SHIFT = 0.5
 
x_min = max_min[0][0][0]
x_max = max_min[0][0][0]
 
y_min = max_min[0][0][1]
y_max = max_min[0][0][1]
 
def shift(val):
  if val == 0:
    return SHIFT
  else:
    return abs(val) * SHIFT
 
for root in max_min:
  x_min = min(x_min, root[0][0])
  y_min = min(y_min, root[0][1])
 
  x_max = max(x_max, root[0][0])
  y_max = max(y_max, root[0][1])

x_min = 0.9
x_max = 2.1
y_min = 0.9
y_max = 2.1
 
x_vals = np.linspace(float(x_min), float(x_max), 1000)
y_vals = np.linspace(float(y_min), float(y_max), 1000)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
 
lam_f = lambdify((x, y), f(x, y), modules=['numpy'])
z_vals = lam_f(x_vals, y_vals)
z_mesh = lam_f(x_mesh, y_mesh)
 
fig = plt.figure(figsize=(10,8))
 
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, cmap = 'coolwarm')

ax.scatter(1, 2, lam_f(1, 2), marker='o', c='red', s=100)
ax.scatter(1, 1, lam_f(1, 1), marker='o', c='blue', s=100)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

x_z = np.linspace(1, 2, 1000)
y_z = np.linspace(1, 2, 1000)
x_z, y_z = np.meshgrid(x_z, y_z)

ax.plot_surface(x_z, y_z, np.zeros((x_z.shape[0], x_z.shape[1])), alpha=0.6, cmap='viridis')

l1_x = lambda y: 1
l2_x = lambda y: 2

l1_y = lambda x: 1
l2_y = lambda x: 2

x_l = np.linspace(1, 2, 10000)
y_l1 = l1_y(x_l)
ax.scatter(x_l, y_l1, lam_f(x_l, y_l1), c='black', s=1)
y_l2 = l2_y(x_l)
ax.scatter(x_l, y_l2, lam_f(x_l, y_l2), c='black', s=1)

y_l = np.linspace(1, 2, 10000)
x_l1 = l1_x(y_l)
ax.scatter(x_l1, y_l, lam_f(x_l1, y_l), c='black', s=1)
x_l2 = l2_x(y_l)
ax.scatter(x_l2, y_l, lam_f(x_l2, y_l), c='black', s=1)

ax.view_init(30, 170)
 
plt.show()
