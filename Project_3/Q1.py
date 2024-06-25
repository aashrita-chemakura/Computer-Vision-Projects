import numpy as np

def compute_camera_pose(img_pts, world_pts):
    n = img_pts.shape[0]

    # Construct the design matrix
    A = np.empty((2*n, 12))
    A.fill(0)

    for i in range(n):
        X, Y, Z = world_pts[i]
        u, v = img_pts[i]
        A[2*i, :] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1, :] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    # Solve for the camera matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    cam_mat = Vt[-1].reshape((3, 4))

    # Decompose P into intrinsic_mat, R, and t matrices using Gram-Schmidt process
    rot_mat, intrinsic_mat = np.linalg.qr(cam_mat[:3,:3])
    rot_mat = rot_mat.T
    intrinsic_mat = intrinsic_mat.T
    transl_v = np.linalg.solve(-cam_mat[:3,:3], cam_mat[:,3])

    # Compute reprojection error for each point
    error = np.zeros(n)
    for i in range(n):
        X, Y, Z = world_pts[i]
        u, v = img_pts[i]

        # Compute reprojected 2D coordinates
        x_proj = cam_mat @ np.array([X,Y,Z,1])
        x_proj /= x_proj[2]

        # Compute reprojection error
        error[i] = np.sqrt((u - x_proj[0])**2 + (v - x_proj[1])**2)

    return intrinsic_mat, rot_mat, transl_v, cam_mat, error

# Define image and world points
img_pts = np.array([[757, 213], [758, 415], [758, 686],[759, 966],
                    [1190, 172], [329, 1041], [1204, 850], [340, 159]])

world_pts = np.array([[0, 0, 0], [0, 3, 0], [0, 7, 0], [0, 11, 0], 
                      [7, 1, 0], [0, 11, 7], [7, 9, 0], [0, 1, 7]])

# Test the function
intrinsic_mat, rot_mat, transl_v, cam_mat, error = compute_camera_pose(img_pts, world_pts)

print("Camera Pose Matrix P:")
print(cam_mat)

print("\nIntrinsic matrix:")
print(intrinsic_mat)
print("\nRotation matrix:")
print(rot_mat)
print("\nTranslation vector:")
print(transl_v)
print("\nReprojection error for each point:")
print(error)