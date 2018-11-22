
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
link_lengths = {g:4, i:2, o:2, f:4} # Should be immutable
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
    print("Input is crank")
    lim_alpha = np.asarray([0, 2*np.pi])
else:
    lim_alpha = np.sort([float(lim_alpha[0]), float(lim_alpha[1])])
    if (np.abs(lim_alpha[0] - 0) < 1e-15) & (np.abs(lim_alpha[1] - np.pi) < 1e-15):
        flag_input_is_crank = True
        print("Input is crank")
        lim_alpha = np.asarray([0, 2*np.pi])

if sp.im(lim_beta[0]) != 0 or sp.im(lim_beta[1]) != 0:
    flag_output_is_crank = True
    print("Output is crank")
    lim_beta = np.asarray([0, 2*np.pi])
else:
    lim_beta = np.sort([float(lim_beta[0]), float(lim_beta[1])])
    if (np.abs(lim_beta[0] - 0) < 1e-15) & (np.abs(lim_beta[1] - np.pi) < 1e-15):
        flag_output_is_crank = True
        print("Output is crank")
        lim_beta = np.asarray([0, 2*np.pi])
    
lim_alpha = [float(elem) for elem in lim_alpha]
lim_beta = [float(elem) for elem in lim_beta]

# lim_alpha = np.sort([sol_alpha[i] if not(flag_input_is_crank) else (i-1)*np.pi for i in (1, 3)])
# lim_beta = np.sort([sol_beta[i] if not(flag_output_is_crank) else i*np.pi for i in (0, 2)])
# lim_alpha, lim_beta

#### Equation generation
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
# eq_input = 2*sp.pi*t + sp.pi/2
# eq_input = (lim_alpha[1] + lim_alpha[0])/2 - (lim_alpha[1] - lim_alpha[0])/2*sp.cos(2*sp.pi*t)
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

# Solve equations symbolically at a given time instant
def state_solver2(t):
    subs_xy = [x0(t), y0(t), x2, y2]
    flag_calc_sols = False
    eqns = sp.simplify(sp.expand(eqs.subs(subs_xy)))
    angle_i = sp.atan2(subs_xy[1][1], subs_xy[0][1])
    angle_g = sp.atan2(subs_xy[3][1], subs_xy[2][1])
    angle_ig = angle_i - angle_g
    if (np.abs(angle_ig) < 1e-14):
        # y1_tmp_eq = [y[1], (const_pars[l[2]]+const_pars[g])*sp.sin(angle_g)]
        if flag_output_is_crank:
            y1_sol = (const_pars[l[2]]+const_pars[g])*sp.sin(angle_g).evalf()
            x1_sol = (const_pars[l[2]]+const_pars[g])*sp.cos(angle_g).evalf()
            y1_sols = [y1_sol, y1_sol]
            x1_sols = [x1_sol, x1_sol]
        else:
            flag_calc_sols = True
            y1_tmp_ex = [y[1], (const_pars[l[2]]+const_pars[g])*sp.sin(angle_g)]
    elif (np.abs(angle_ig-np.pi) < 1e-14):
        # y1_tmp_eq = [y[1], (-const_pars[l[2]]+const_pars[g])*sp.sin(angle_g)]
        if flag_output_is_crank:
            y1_sol = (-const_pars[l[2]]+const_pars[g])*sp.sin(angle_g).evalf()
            x1_sol = (-const_pars[l[2]]+const_pars[g])*sp.cos(angle_g).evalf()
            y1_sols = [y1_sol, y1_sol]
            x1_sols = [x1_sol, x1_sol]
        else:
            flag_calc_sols = True
            y1_tmp_ex = [y[1], (-const_pars[l[2]]+const_pars[g])*sp.sin(angle_g)]
    else:
        flag_calc_sols = True
        y1_tmp_ex = [y[1], sp.solve(eqns[1]-eqns[0], y[1])[0]]
    
    if flag_calc_sols:
        x1_tmp_eq = sp.simplify(eqns[0].subs(*y1_tmp_ex))
        x1_sols = sp.solve(x1_tmp_eq)
        y1_sols = [sp.simplify(y1_tmp_ex[1].subs(x[1], x1_sols[i])) for i in (0, 1)]
    
    phi1_sols = [sp.atan2(y1_sols[i]-y2[1], x1_sols[i]-x2[1]).evalf() for i in (0, 1)]
    phi1_sols = [phi1_sols[i] if phi1_sols[i] >= 0 else phi1_sols[i] + 2*np.pi for i in (0, 1)]
    return [[subs_xy[0][1], subs_xy[1][1]], x1_sols, y1_sols, phi1_sols]

range_t = np.arange(0, 1, 0.01)
steps = len(range_t)
x_coords = np.zeros([steps, 4])
y_coords = np.zeros([steps, 4])
phi1_vals = np.zeros([steps, 1])
x_coords[:, 3] = x2[1]
y_coords[:, 3] = y2[1]
# Solve for the 0th time
init_coords = state_solver2(range_t[0])
x_coords[0, 1:3] = [init_coords[0][0], init_coords[1][1]]
y_coords[0, 1:3] = [init_coords[0][1], init_coords[2][1]]
phi1_vals[0] = init_coords[3][1]

for i in range(1, steps):
    print("\rt={}".format(range_t[i]), end="")
    coords = state_solver2(range_t[i])
    j = 1
    x_coords[i, 1] = coords[0][0]
    y_coords[i, 1] = coords[0][1]
    
    if flag_output_is_crank:
        # flag_2pi = False
        # if 2*np.pi - phi1_vals[i-1] < 0.2:
        #     flag_2pi = True
        #     coords[3][0] += np.pi
        #     coords[3][1] += np.pi
        angle_diff_0 = coords[3][0] - phi1_vals[i-1]
        angle_diff_1 = coords[3][1] - phi1_vals[i-1]
        if (angle_diff_0 > 0) & (angle_diff_1 > 0):
            # Select the closest
            # phi1_vals[i] = coords[3][0] if angle_diff_0 < angle_diff_1 else coords[3][1]
            j = 0 if angle_diff_0 < angle_diff_1 else 1
        else:
            # Select the angle with positive difference
            # phi1_vals[i] = coords[3][0] if angle_diff_0 > angle_diff_1 else coords[3][1]
            j = 0 if angle_diff_0 > angle_diff_1 else 1
    x_coords[i, 2] = coords[1][j]
    y_coords[i, 2] = coords[2][j]
    phi1_vals[i] = coords[3][j]

print("\nCreating animation")

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

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
                              interval=10, blit=True, init_func=init)

# ani.save('crank-crank.mp4', writer=writer)
plt.show()
