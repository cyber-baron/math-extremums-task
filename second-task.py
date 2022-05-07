import sympy
from sympy import Lambda, log, Derivative, diff, Rational, simplify, linsolve, lambdify, exp, solve, ask, Q, I, S
from sympy.abc import x, y
from sympy.plotting import plot3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

f = Lambda((x, y), x * y + 50/x + 20/y)

f_dx = Lambda((x, y), diff(f(x, y), x))
f_2dx = Lambda((x, y), diff(f_dx(x, y), x))

f_dy = Lambda((x, y), diff(f(x, y), y))
f_2dy = Lambda((x, y), diff(f_dy(x, y), y))

f_dxdy = Lambda((x, y), diff(f_dx(x, y), y))

grad_0 = solve([f_dx(x, y), f_dy(x, y)], (x, y)

roots = list()

for root in grad_0:
  roots.append(root)

max_min = list()

for root in roots:
  A = f_2dx(root[0], root[1])
  print(A)
  B = f_dxdy(root[0], root[1])
  print(B)
  C = f_2dy(root[0], root[1])
  print(C)

  expr = A * C - B**2
  print(expr)

  if ask(Q.rational(expr)) == True:
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

SHIFT = 0.75

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

x_min -= shift(x_min)
x_max += shift(x_max)

y_min -= shift(y_min)
y_max += shift(y_max)


x_vals = np.linspace(float(x_min), float(x_max), 1000)
y_vals = np.linspace(float(y_min), float(y_max), 1000)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

lam_f = lambdify((x, y), f(x, y), modules=['numpy'])
z_vals = lam_f(x_vals, y_vals)
z_mesh = lam_f(x_mesh, y_mesh)

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, cmap = 'coolwarm')

for root in max_min:
  if root[2] == 1:
    ax.scatter(float(root[0][0]), float(root[0][1]), float(root[1]), marker='o', c='red', s=100)
  elif root[2] == -1:
    ax.scatter(float(root[0][0]), float(root[0][1]), float(root[1]), marker='o', c='blue', s=100)

ax.view_init(20, 110)

plt.show()
