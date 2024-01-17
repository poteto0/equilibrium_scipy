import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy.linalg as LA
import scipy.linalg as la
import math
from scipy import optimize
from scipy.integrate import odeint
from autograd import jacobian

class equilibrium:
  def __init__(self, func):
    self.func = func

  def draw_phase_diagram(self, y1, y2, y1_base=1, colors=["green", "red"]):
    Y1, Y2 = np.meshgrid(y1, y2)
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape

    for i in range(NI):
      for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = self.func([x, y])
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
    
    plt.quiver(Y1/y1_base, Y2, u, v, color='grey')
    plt.contour(Y1/y1_base, Y2, u, levels=[0], colors=colors[0])
    plt.contour(Y1/y1_base, Y2, v, levels=[0], colors=colors[1])
  
  def show_plot(self):
    plt.show()

  def ode(self, y0 = [1.1, 0.3], y1_base=1, t_length=100000):
    tspan = np.linspace(0, 10000, t_length)
    ys = odeint(self.func, y0, tspan)
    plt.plot(ys[:,0]/y1_base, ys[:,1], 'b--', color = 'k') # path
  
  def multi_ode(self, y0 = [1.1, 0.3], n=6):
    tspan = np.linspace(0, 100, 1000)
    y0_reshaped = y0.reshape(n*3,)
    return odeint(self.func, y0_reshaped, tspan)
  
  def multi_equilibrium_analyze_and_plot(self, x_list = [[1.0, 0.0]], method = 'hybr'):
    for x in x_list:
      self.equilibrium_analyze_and_plot(x, method)

  def equilibrium_analyze_and_plot(self, x0 = [1.0, 0.0], method = 'hybr'):
    (point, is_stable) = self.equilibrium_analyze(x0, method)
    self.plot_equilibrium_point(point, is_stable)
  
  def equilibrium_analyze(self, x0 = [1.0, 0.0], method = 'hybr'):
    point, jacobi = self.root(x0, method)
    print(point)
    return (point, self.is_stable(self.calc_eigen(jacobi)))
  
  def plot_equilibrium_point(self, point, is_stable):
    color = 'k' if is_stable else 'w'
    plt.scatter(point[0]/14, point[1], marker = 'o', edgecolors = 'k', color = color, zorder=2)

  def root(self, x0 = [1.0, 0.0], method = 'hybr'):
    result = optimize.root(self.func, x0, method=method, options=dict(eps=1e-6))
    R = self.__reshape_upper_triangle_matrix(result.fjac.shape, result.r)
    return (result.x, np.dot(result.fjac.T, R))

  # 上三角形行列を直す
  def __reshape_upper_triangle_matrix(self, target_shape, r_list):
    R = np.zeros(target_shape)
    i, j = 0, 0
    for r_i in r_list:
      R[i][j] = r_i
      j += 1
      if j == R.shape[0]:
        i += 1
        j = i
    return R
  
  def calc_eigen(self, jacobi):
    return LA.eig(jacobi)[0]
  
  def is_stable(self, eigen):
    for e in eigen:
      if e.real >= 0:
        return False
    return True