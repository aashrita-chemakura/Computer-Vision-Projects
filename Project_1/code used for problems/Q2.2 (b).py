import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data1 = np.loadtxt("given/pc1.csv", dtype = float, delimiter=",")
data2 = np.loadtxt("given/pc2.csv", dtype = float, delimiter=",")

A1 = np.array(data1[:,0])
B1 = np.array(data1[:,1])
C1 = np.array(data1[:,2])

size1 = data1.shape[0]
size2 =data2.shape[0]

A2 = np.array(data2[:,0])
B2 = np.array(data2[:,1])
C2 = np.array(data2[:,2])


A1_mean = np.mean(data1[:,0])
B1_mean = np.mean(data1[:,1])
C1_mean = np.mean(data1[:,2])

A2_mean = np.mean(data2[:,0])
B2_mean = np.mean(data2[:,1])
C2_mean = np.mean(data2[:,2])

X1 = np.vstack([(A1).reshape(1,size1)-A1_mean, (B1).reshape(1,size1)-B1_mean, (C1).reshape(1,size1)-C1_mean]).T
X2 = np.vstack([(A2).reshape(1,size2)-A2_mean, (B2).reshape(1,size2)-B2_mean, (C2).reshape(1,size2)-C2_mean]).T


def SVD(X):

    XT = X.T
    Product1 = np.dot(X,XT)
    eval_U1, evec_U1 = np.linalg.eig(Product1)
    sort_temp = eval_U1.argsort()[::-1]
    eval_U1.sort()
    evec_U1 = evec_U1[:, sort_temp]

    Product2 = np.dot(XT,X)
    eval_V1, evec_V1 = np.linalg.eig(Product2)
    eval_V1.sort()
    evec_V1 = evec_V1[:, eval_V1.argsort()[::-1]]

    sig1 = np.zeros_like(X)
    Solu_1= np.sqrt(eval_V1)
    diagnol_U = np.diag(Solu_1)
    sig1[:diagnol_U.shape[0], :diagnol_U.shape[1]] = diagnol_U

    return evec_U1, sig1, evec_V1


def plane_normal(X):
    u, sigma, v = SVD(X)
    normal = v[:,v.shape[1]-1]
    d = normal[0]*A1_mean + normal[1]*B1_mean+ normal[2]*C1_mean
    return normal, d

_, dist1 = plane_normal(X1)
_, dist2 = plane_normal(X2)

def ransac_fit(data, iter, thresh, dist):
    best_fit = None
    num_points = data.shape[0]
    inlier_c = num_points -100
    i = 0
    while(i<iter):
        random_indices = list(np.random.choice(num_points, 4, replace=False))
        sample_points = np.take(data, random_indices, axis=0)
        normal = plane_normal(sample_points)[0]
        inliers = np.abs(np.dot(data, normal) + dist) / np.linalg.norm(normal)

        n_inliers = np.where(inliers < thresh)
        n_inliers = np.array([n_inliers]).shape[2]

        if n_inliers > inlier_c:
            best_fit = normal
            total_inliers = inliers
        i+=1

    return best_fit, total_inliers

normal1, inliers1 = ransac_fit(data1, 10000, 1, dist1)
normal2, inliers2 = ransac_fit(data2, 10000, 1, dist2)

a1, b1, c1 = normal1
a2, b2, c2 = normal2

d1 = a1*A1_mean + b1*B1_mean+ c1*C1_mean
d2 = a2*A2_mean + b2*B2_mean+ c2*C2_mean

print("pc1 equation of the plane: ",a1,'*x+',b1,'*y+',c1,'*z+',d1,'=0')
print("No. of Inliers: ", len(inliers1))

print("pc2 equation of the plane: ",a2,'*x+',b2,'*y+',c2,'*z+',d2,'=0')
print("No. of Inliers: ", len(inliers2))


fig1 = plt.figure(figsize = (40, 7))
ax1 = fig1.add_subplot(131,projection='3d')
ax2 = fig1.add_subplot(132,projection='3d')
ax3 = fig1.add_subplot(133,projection='3d')

ax1.scatter3D(A1,B1,C1)
ax3.scatter3D(A1,B1,C1)
ax2.scatter3D(A2,B2,C2)
ax3.scatter3D(A2,B2,C2)

ax1.plot_trisurf(data1[:,0],data1[:,1], (d1 - a1*data1[:,0] - b1*data1[:,1])/(c1))
ax2.plot_trisurf(data2[:,0],data2[:,1], (d2 - a2*data2[:,0] - b2*data2[:,1])/(c2))
ax3.plot_trisurf(data1[:,0],data1[:,1], (d1 - a1*data1[:,0] - b1*data1[:,1])/(c1))
ax3.plot_trisurf(data2[:,0],data2[:,1], (d2 - a2*data2[:,0] - b2*data2[:,1])/(c2))

ax1.set_title("pc1")
ax2.set_title("pc2")
ax3.set_title("pc1 & pc2")

plt.show()

