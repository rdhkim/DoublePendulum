import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# using sympy outline symbols to be used in equations
t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

# define theta 1 & 2 as symbols and functions of time
theta1, theta2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
theta1, theta2 = theta1(t), theta2(t)

# define derivatives and second derivatives of theta 1 & 2
theta1_d = smp.diff(theta1, t)
theta2_d = smp.diff(theta2, t)
theta1_dd = smp.diff(theta1_d, t)
theta2_dd = smp.diff(theta2_d, t)

# define x1, y1, x2, and y2 based on theta 1 & 2
x1 = L1 * smp.sin(theta1)
y1 = -L1*smp.cos(theta1)
x2 = L2*smp.sin(theta1)+L2*smp.sin(theta2)
y2 = -L1*smp.cos(theta1)-L2*smp.cos(theta2)

# use x1, y1, x2, and y2 to define kinetic and potential energy for mass. 
# used to obtain Lagrangian

# Kinetic
T1 = 1/2 * m1 * (smp.diff(x1, t)**2 + smp.diff(y1,t)**2)
T2 = 1/2 * m2 * (smp.diff(x2, t)**2 + smp.diff(y2,t)**2)
T = T1+T2

# Potential
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2

# Lagrangian
L = T-V

# obtain Lagrange's equations
LE1 = smp.diff(L, theta1) - smp.diff(smp.diff(L, theta1_d), t).simplify()
LE2 = smp.diff(L, theta2) - smp.diff(smp.diff(L, theta2_d), t).simplify()

# Solve Lagranges equation where LE1 & LE2 are both equal to 0
solve = smp.solve([LE1, LE2], (theta1_dd, theta2_dd), 
                                simplify=False, rational=False)

# convert second order ODEs into 4 first order ODEs 
# convert symbolic expressoins to numerical functions using lambdify
dz1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,theta1,theta2,theta1_d,theta2_d), 
                        solve[theta1_dd])
dz2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,theta1,theta2,theta1_d,theta2_d), 
                        solve[theta2_dd])
dtheta1dt_f = smp.lambdify(theta1_d, theta1_d)
dtheta2dt_f = smp.lambdify(theta2_d, theta2_d)

# define function to solve for derivative of speed and time
def dSdt(S, t, g, m1, m2, L1, L2):
    theta1, z1, theta2, z2 = S
    return [
        dtheta1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, theta1, theta2, z1, z2),
        dtheta2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, theta1, theta2, z1, z2),
    ]

# solve system of ODEs using scipy and odeint method
t = np.linspace(0, 40, 1001)
g = 9.81
m1 = 2
m2 = 1
L1 = 2
L2 = 1
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t, args=(g,m1,m2,L1,L2))

# obtain theta1(t) and theta2(t) from the answer
theta1 = ans.T[0]
theta2 = ans.T[1]

plt.plot(t, theta2)

def get_x1y1x2y2(t, theta1, theta2, L1, L2):
    return (
        L1 * np.sin(theta1),
        -L1 * np.cos(theta1),
        L1 * np.sin(theta1) + L2 * np.sin(theta2),
        -L1 * np.cos(theta1) - L2 * np.cos(theta2)
    )
x1, y1, x2, y2 = get_x1y1x2y2(t, ans.T[0], ans.T[2], L1, L2)

# make the animation
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ln1, = plt.plot([],[], 'ro--', lw=3, markersize=8)
ln2, = ax.plot([],[], 'ro--', markersize=8, alpha=0.05, color='cyan')
ln3, = ax.plot([],[], 'ro--', markersize=8, alpha=0.05, color='cyan')
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pen.gif', writer='pillow', fps=25)
