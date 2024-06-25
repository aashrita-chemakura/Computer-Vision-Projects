import numpy as np
import cv2

# Define calibration target parameters
checkbd_size = (9, 6)
sq_size = 21.5  # in mm

# Prepare object points
obj_pts = np.zeros((np.prod(checkbd_size), 3), dtype=np.float32)
obj_pts[:, :2] = np.indices(checkbd_size).T.reshape(-1, 2)
obj_pts *= sq_size

# Initialize image and object points lists
img_points = []  # 2D points in image plane
obj_points = []  # 3D points in real world space

# Load calibration images
for i in range(1, 14):
    img = cv2.imread(f'Given/calib_img{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkbd_size, None)

    if ret:
        # Refine corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners)
        obj_points.append(obj_pts)

        # Draw and display corners
        cv2.drawChessboardCorners(img, checkbd_size, corners, ret)
        cv2.imshow(f"Calibration Image", cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3))))
        cv2.imwrite(f"Results/img_{i}.jpg", img)
        cv2.waitKey(50)
    else:
        print(f"No corners found in image {i}")

cv2.destroyAllWindows()

# Compute camera calibration
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print camera matrix and distortion coefficients
print("Camera Matrix (K):")
print(K)
print("\nDistortion Coefficients:")
print(dist)

# Compute and print reprojection error
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    mean_error += error
print(f"\nMean Reprojection Error: {mean_error/len(obj_points)} pixels")
