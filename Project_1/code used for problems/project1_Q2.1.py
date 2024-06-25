import numpy as np

# 2.1. (a) : Computing covariance matrix

#Loading data from file
data = np.loadtxt("given/pc1.csv", delimiter= ',', dtype = float)

A1 = data[:,0]
B1 = data[:,1]
C1 = data[:,2]

A1_mean  = np.mean(A1)
B1_mean = np.mean(B1)
C1_mean = np.mean(C1)

A_1 = (A1 - A1_mean)**2
B_1 = (B1 - B1_mean)**2
C_1 = (C1 - C1_mean)**2

Var_A = np.sum(A_1)/len(A1)
Var_B = np.sum(B_1)/len(B1)
Var_C = np.sum(C_1)/len(C1)

a1 = (A1 - A1_mean)
b1 = (B1 - B1_mean) 
c1 = (C1 - C1_mean)

cov_AB = np.dot(a1, b1)/(len(A1))
cov_BC = np.dot(b1, c1)/(len(A1))
cov_AC = np.dot(a1, c1)/(len(A1))

Matrix = np.matrix([[Var_A, cov_AB, cov_AC], [cov_AB, Var_B, cov_BC], [cov_AC,cov_BC, Var_C]])
print("The covariance matrix is: \n", Matrix)

# 2.1. (b) Computing magnitude and direction of surface normal 

e_value , e_vector = np.linalg.eig(Matrix)
min_eval = np.argmin(e_value)
Surfnorm = e_vector[:, min_eval]
Mag_surfnorm = np.sqrt(Surfnorm[0]**2 + Surfnorm[1]**2 + Surfnorm[2]**2)
print("\n The surface normal is: \n", Surfnorm)
print("\nThe magnitude of the surface normal is:\n", Mag_surfnorm)
