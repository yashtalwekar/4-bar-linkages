import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
import matplotlib.animation as animation
# from sympy.parsing.sympy_parser import parse_expr
# sp.init_printing()

from sympy.vector import CoordSys3D
N = CoordSys3D('N')

# http://dynref.engr.illinois.edu/aml.html
# Refer above link for notation
g, i, o, f = sp.symbols('g i o f')
alpha, beta, theta_g = sp.symbols('alpha beta theta_g')
subs_x, subs_y = sp.symbols('x y')

# Origin at intersection of ground and input links
# Input pivot A, Output pivot B
A = N.origin.locate_new('A', i*sp.cos(alpha)*N.i + i*sp.sin(alpha)*N.j + 0*N.k)
B = N.origin.locate_new('B', (g*sp.cos(theta_g) + o*sp.cos(beta))*N.i + (g*sp.sin(theta_g) + o*sp.sin(beta))*N.j + 0*N.k)

# A.express_coordinates('N')

# Order for link specifications:
# ground, input, output, floating
link_lengths = {g:4, i:3, o:2, f:4} # Should be immutable
# angle_input = 90.0
angle_ground = 0

# We don't know the following initially
flag_input_is_crank = False
flag_output_is_crank = False

# Substitutions for angles to convert solution in range (0, pi) to (0, 2*pi)
z = sp.symbols('z')
subs_list_alpha = [alpha, alpha, alpha - sp.pi, alpha]
subs_list_beta = [sp.pi - beta, sp.pi - beta, sp.pi - beta, 2*sp.pi - beta]

# Limit cases:
# 1: Input + floating make a longer link
# 2: Output + floating make a longer link
# 3: Floating - input make a shorter link
# 4: Floating - output make a shorter link

# Note that limits for alpha will be obtained in the cases 2 and 4, and for beta in 1 and 3
# If the limits are imaginary, it means that that link is a crank
# If real, then the link is a rocker between those limits

# Equations to be solved for alpha
mod_eqns_alpha = [\
                (g - (i+f)*sp.cos(z))**2 + ((i+f)*sp.sin(z))**2 - o**2,\
                (g - i*sp.cos(z))**2 + (i*sp.sin(z))**2 - (o+f)**2,\
                (g - (f-i)*sp.cos(z))**2 + ((f-i)*sp.sin(z))**2 - o**2,\
                (g - i*sp.cos(z))**2 + (i*sp.sin(z))**2 - (f-o)**2\
                ]

# Equations to be solved for beta
mod_eqns_beta = [\
                (g - o*sp.cos(z))**2 + (o*sp.sin(z))**2 - (i+f)**2,\
                (g - (o+f)*sp.cos(z))**2 + ((o+f)*sp.sin(z))**2 - i**2,\
                (g - o*sp.cos(z))**2 + (o*sp.sin(z))**2 - (f-i)**2,\
                (g - (f-o)*sp.cos(z))**2 + ((f-o)*sp.sin(z))**2 - i**2\
                ]

# Simplify equations
eqns_alpha = [sp.simplify(sp.expand(mod_eqns_alpha[i])) for i in range(0, 4)]
eqns_beta = [sp.simplify(sp.expand(mod_eqns_beta[i])) for i in range(0, 4)]

# Substitute link lengths
temp_alpha = [eqns_alpha[i].subs(link_lengths) for i in range(0, 4)]
temp_beta = [eqns_beta[i].subs(link_lengths) for i in range(0, 4)]

# sol_alpha = [sp.acos(sp.solve(eqns_alpha[i].subs(link_lengths).evalf(), sp.cos(alpha))[0]) for i in range(0, 4)]
# sol_beta = [sp.acos(sp.solve(eqns_beta[i].subs(link_lengths).evalf(), sp.cos(beta))[0]) for i in range(0, 4)]

# Solve equations for z as that is the only unknown in the equation
sol_alpha = [sp.acos(sp.solve(eqns_alpha[i].subs(link_lengths).evalf(), sp.cos(z))[0]) for i in range(0, 4)]
sol_beta = [sp.acos(sp.solve(eqns_beta[i].subs(link_lengths).evalf(), sp.cos(z))[0]) for i in range(0, 4)]

# Retrieve corrections specified earlier
corrected_sol_alpha = [sol_alpha[0], sol_alpha[1], sp.pi + sol_alpha[2], sol_alpha[3]]
corrected_sol_beta = [sp.pi - sol_beta[0], sp.pi - sol_beta[1], sp.pi - sol_beta[2], 2*sp.pi - sol_beta[3]]

# Save the limits: alpha limits are obtained from equations 2 and 4, and beta limits from 1 and 3
lim_alpha = [corrected_sol_alpha[i] for i in (1, 3)]
lim_beta = [corrected_sol_beta[i] for i in (0, 2)]

# Check if limits are imaginary and if true, flag the link as a crank and set the limits 0 to 2*pi
if sp.im(lim_alpha[0]) != 0 or sp.im(lim_alpha[1]) != 0:
    flag_input_is_crank = True
    lim_alpha = np.asarray([0, 2*np.pi])
else:
    lim_alpha = np.sort([float(lim_alpha[0]), float(lim_alpha[1])])
    if (np.abs(lim_alpha[0] - 0) < 1e-15) & (np.abs(lim_alpha[1] - np.pi) < 1e-15):
        flag_input_is_crank = True
        lim_alpha = np.asarray([0, 2*np.pi])

if sp.im(lim_beta[0]) != 0 or sp.im(lim_beta[1]) != 0:
    flag_output_is_crank = True
    lim_beta = np.asarray([0, 2*np.pi])
else:
    lim_beta = np.sort([float(lim_beta[0]), float(lim_beta[1])])
    if (np.abs(lim_beta[0] - 0) < 1e-15) & (np.abs(lim_beta[1] - np.pi) < 1e-15):
        flag_output_is_crank = True
        lim_beta = np.asarray([0, 2*np.pi])
    
lim_alpha = [float(elem) for elem in lim_alpha]
lim_beta = [float(elem) for elem in lim_beta]

# lim_alpha = np.sort([sol_alpha[i] if not(flag_input_is_crank) else (i-1)*np.pi for i in (1, 3)])
# lim_beta = np.sort([sol_beta[i] if not(flag_output_is_crank) else i*np.pi for i in (0, 2)])
# lim_alpha, lim_beta

#### Equation generation for solving the configuration
# ground link length and angle
g, theta = sp.symbols('g theta')
# For other links
# Input link origin at coordinate system origin
# 1: Pivot A; 2: Pivot B; 3: Fixed end of output link
# x, y coordinates; angle phi wrt to ground link; lengths l
x = sp.symbols('x1:4')
y = sp.symbols('y1:4')
phi = sp.symbols('phi1:4')
l = sp.symbols('l1:4')
t = sp.Symbol('t')

const_pars = {l[0]: link_lengths[i], l[1]: link_lengths[f], l[2]:link_lengths[o], g:link_lengths[g], theta: angle_ground}

# Sample input
if flag_input_is_crank:
	eq_input = 2*sp.pi*t
else:
	eq_input = (lim_alpha[1] + lim_alpha[0])/2 - (lim_alpha[1] - lim_alpha[0])/2*sp.cos(2*sp.pi*t)
func_input = lambdify(t, eq_input, 'numpy')
x0 = lambda t: [x[0], const_pars[l[0]]*np.cos(func_input(t))]
y0 = lambda t: [y[0], const_pars[l[0]]*np.sin(func_input(t))]
x2 = [x[2], const_pars[g]*np.cos(const_pars[theta])]
y2 = [y[2], const_pars[g]*np.sin(const_pars[theta])]

# Equations to be solved
eqs = sp.Matrix([\
                (x[1]-x[0])**2 + (y[1]-y[0])**2 - l[1]**2,\
                (x[2]-x[1])**2 + (y[2]-y[1])**2 - l[2]**2\
                ])

eqs = eqs.subs(const_pars)

# state = sp.Matrix([elem[j] for elem in [[phi[i], x[i], y[i]] for i in range(0, 3)] for j in range(0, 3)][1:])
state = sp.Matrix([x[1], y[1]])

# Define Newton-Raphson based solver
def newtont(fun, x):
    n = x.shape[0]
    epsil = 1e-5 * max(1, np.linalg.norm(x))
    pert = np.identity(n) * epsil
    iter = 0
    nmax = 600
    conv = 1
    D = np.zeros([n, n])

    ee = fun(*x)

    while (np.linalg.norm(ee)*max(1, np.linalg.norm(x))>1e-10) and (iter<nmax):
        iter += 1
        for k in range(0, n):
            D[:, k] = ((fun(*(x + pert[:, k])) - ee)/epsil)[:, 0]

        x = x - np.linalg.solve(D, ee)[:, 0]
        ee = fun(*x)
    print(iter, "Iterations that took")
    if iter is nmax or abs(x) is np.inf:
        conv = 0
        print("Did not converge")
    return [np.asarray(x), conv]


def state_solver(t, guess):
    subs_xy = [x0(t), y0(t), x2, y2]
    eqns = lambdify(state, eqs.subs(subs_xy), 'numpy')
    x1y1 = newtont(eqns, guess)[0]
    return [subs_xy[0][1], x1y1[0], subs_xy[1][1], x1y1[1]]

# Time range
range_t = np.arange(0, 1, 0.01)
steps = len(range_t)
x_coords = np.zeros([steps, 4])
y_coords = np.zeros([steps, 4])
x_coords[:, 3] = x2[1]
y_coords[:, 3] = y2[1]
# Guess for initial state
init_coords = np.asarray([6, 0])
# Solve for initial state
init_coords = state_solver(range_t[0], init_coords)
x_coords[0, 1:3] = init_coords[:2]
y_coords[0, 1:3] = init_coords[2:]
x_coords[0], y_coords[0]

# Solve the system at each time step
for i in range(1, steps):
    coords = state_solver(range_t[i], np.asarray([x_coords[i-1, 2], y_coords[i-1, 2]]))
    x_coords[i, 1:3] = coords[:2]
    y_coords[i, 1:3] = coords[2:]

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
ax.grid()
ax.plot(0, 0)

line, = ax.plot([], [], 'bo-')

def init():
    line.set_data([], [])
    return line,

def animate(num):
    
    line.set_data(x_coords[num], y_coords[num])
    return line,
    
ani = animation.FuncAnimation(fig, animate, range(0, steps),
                              interval=20, blit=True, init_func=init)

plt.show()

