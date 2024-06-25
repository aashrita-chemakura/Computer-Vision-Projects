import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# (Q.2.2 (a))- Least Squares

# Loading the data
data1 = np.loadtxt("given/pc1.csv", dtype = float, delimiter=",")
data2 = np.loadtxt("given/pc2.csv", dtype = float, delimiter=",")

A1 = np.array(data1[:,0])
B1 = np.array(data1[:,1])
C1 = np.array(data1[:,2])

size1 = data1.shape[0]

A2 = np.array(data2[:,0])
B2 = np.array(data2[:,1])
C2 = np.array(data2[:,2])

size2 = data2.shape[0]

D1 = np.vstack([A1.reshape(1,size1),B1.reshape(1,size1),np.ones((1,size1))]).T
P1 = np.linalg.inv(np.dot(D1.T,D1))
Q1 = np.dot(P1,D1.T)
S_1 = np.dot(Q1,C1)
a1 = S_1[0]
b1 = S_1[1]
c1 = S_1[2]

D2 = np.vstack([A2.reshape(1,size2),B2.reshape(1,size2),np.ones((1,size2))]).T
P2 = np.linalg.inv(np.dot(D2.T,D2))
Q2 = np.dot(P2,D2.T)
S_2 = np.dot(Q2,C2)

a2 = S_2[0]
b2 = S_2[1]
c2 = S_2[2]

print("Equation of the plane for data set 2 is :",a2,"*x +",b2,"*y +", c2 ,"\n")

fig = plt.figure(figsize = (10, 7))
axis1 = fig.add_subplot(121,projection='3d')
axis2 = fig.add_subplot(122,projection='3d')
axis1.scatter3D(A1,B1,C1)
axis2.scatter3D(A2,B2,C2)

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
plane1 = a1 * x + b1 * y + c1
plane2 = a2 * x + b2 * y + c2

axis1.plot_surface(x, y, plane1)
axis2.plot_surface(x, y, plane2)

# Total Least Squares

A1_mean = np.mean(data1[:,0])
B1_mean = np.mean(data1[:,1])
C1_mean = np.mean(data1[:,2])

A2_mean = np.mean(data2[:,0])
B2_mean = np.mean(data2[:,1])
C2_mean = np.mean(data2[:,2])

X1 = np.vstack([(A1).reshape(1,size1)-A1_mean, (B1).reshape(1,size1)-B1_mean, (C1).reshape(1,size1)-C1_mean]).T
X2 = np.vstack([(A2).reshape(1,size2)-A2_mean, (B2).reshape(1,size2)-B2_mean, (C2).reshape(1,size2)-C2_mean]).T

# --------------------------------------------------- For data set 1 -----------------------------------------------------
X1Trans = X1.T
Product1 = np.dot(X1,X1Trans)
eval_U1, evec_U1 = np.linalg.eig(Product1)
sort_temp = eval_U1.argsort()[::-1]
eval_U1.sort()
evec_U1 = evec_U1[:, sort_temp]

Product2 = np.dot(X1Trans,X1)
eval_V1, evec_V1 = np.linalg.eig(Product2)
eval_V1.sort()
evec_V1 = evec_V1[:, eval_V1.argsort()[::-1]]

sig1 = np.zeros_like(X1)
Solu_1= np.sqrt(eval_V1)
diagnol_U = np.diag(Solu_1)
sig1[:diagnol_U.shape[0], :diagnol_U.shape[1]] = diagnol_U

normal1 = evec_V1[:,evec_V1.shape[1]-1]
n1_x = normal1[0]
n1_y = normal1[1]
n1_z = normal1[2]
s1 = n1_x*A1_mean + n1_y*B1_mean + n1_z*C1_mean

# --------------------------------------------------- For data set 2 -----------------------------------------------------
X2Trans = X2.T
Product_1 = (np.dot(X2,X2Trans))
eval_U2, evec_U2 = np.linalg.eig(Product_1)
eval_U2.sort()
evec_U2 = evec_U2[:, eval_U2.argsort()[::-1]]
Product_2 = np.dot(X2Trans,X2)
eval_V2, evec_V2 = np.linalg.eig(Product_2)
temp_sort = eval_V2.argsort()[::-1]
eval_V2.sort()
evec_V2 = evec_V2[:, temp_sort]

sig2 = np.zeros_like(X2)
Solu_2 = np.sqrt(eval_V2)
diagnolV = np.diag(Solu_2)

sig2[:diagnolV.shape[0], :diagnolV.shape[1]] = diagnolV

normal2 = evec_V2[:,evec_V2.shape[1]-1]
n2_x = normal2[0]
n2_y = normal2[1]
n2_z = normal2[2]
s2 = n2_x*A2_mean + n2_y*B2_mean+ n2_z*C2_mean

# ------------------------------------------------------- Plots ------------------------------------------------------------
fig2 = plt.figure(figsize = (10, 7))
axis1 = fig2.add_subplot(121,projection='3d')
axis2 = fig2.add_subplot(122,projection='3d')
axis1.scatter3D(A1,B1,C1)
axis2.scatter3D(A2,B2,C2)

axis1.plot_trisurf(A1,B1, (s1 - n1_x*A1 - n1_y*B1)/(n1_z))
axis2.plot_trisurf(A2,B2, (s2 - n2_x*A2 - n2_y*B2)/(n2_z))

print("Equation 1: ", n1_x,'*x+',n1_y,'*y+',n1_z,'*z+',s1,'=0')
print("Equation 2: ", n2_x,'*x+',n2_y,'*y+',n2_z,'*z+',s2,'=0')


plt.show()