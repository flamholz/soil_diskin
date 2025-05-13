import numpy as np
from sympy import Matrix, symbols, Symbol, Function, latex
from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel

parCESM = np.loadtxt('compartmentalParameters.txt', skiprows = 1)

tau_1, tau_2, tau_3, r_f, r_s, u_1 = symbols('tau_1 tau_2 tau_3 r_f r_s u_1')

B1 = Matrix([[-1/tau_1, 0, 0],[r_f/tau_1, -1/tau_2, 0], [0, r_s/tau_2, -1/tau_3]])
u1 = Matrix([[u_1], [0], [0]])

n = parCESM.shape[0]

out = np.zeros([n, 5])
for i in range(0, n):
    mat_dict = {tau_1:  parCESM[i,1], tau_2:  parCESM[i,2], tau_3: parCESM[i,3], r_f: parCESM[i,4], r_s: parCESM[i,5]}
    u_dict = {u_1: parCESM[i,6]}
    M1 = LinearAutonomousPoolModel(u = u1.subs(u_dict), B = B1.subs(mat_dict), force_numerical=True)
    out[i, : ] = M1.T_expected_value, M1.A_expected_value, M1.A_quantile(0.05), M1.A_quantile(0.5), M1.A_quantile(0.95)
    print(i+1)

np.savetxt("CESMages.csv", out, delimiter = ",")

print("Ran succesfully!")

