import cv2
import numpy as np
import random

# Function to rescale the images
def rescale(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Computing homography by using RANSAC
def compute_homography(pts1, pts2, n_iterations=1000, thresh=5):
    best_H = None
    max_inliers = 0
    for i in range(n_iterations):
        # Select 4 random points
        chosen_idx = random.sample(range(pts1.shape[0]), 4)
        A = np.zeros((8, 9))
        for j in range(4):
            x1, y1 = pts1[chosen_idx[j]][0]
            x2, y2 = pts2[chosen_idx[j]][0]
            A[j*2, :] = np.array([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
            A[j*2+1, :] = np.array([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
        # Solve linear system using SVD
        _, _, V = np.linalg.svd(A)
        h = V[-1, :] / V[-1, -1]
        H = h.reshape((3, 3))
        # Count inliers
        inliers_ct = 0
        for j in range(pts1.shape[0]):
            p1 = np.array([pts1[j][0][0], pts1[j][0][1], 1])
            p2 = np.array([pts2[j][0][0], pts2[j][0][1], 1])
            transformed_p1 = np.dot(H, p1)
            transformed_p1 /= transformed_p1[2]
            d = np.linalg.norm(transformed_p1 - p2)
            if d < thresh:
                inliers_ct += 1
        # Update best homography
        if inliers_ct > max_inliers:
            best_H = H
            max_inliers = inliers_ct
    return best_H

def feature_matching(img1, img2):
    # Converting the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Using ORB to extract features
    orb = cv2.ORB_create()

    # Defining keypoints and descriptors
    kpi, desi = orb.detectAndCompute(gray1, None)
    kpj, desj = orb.detectAndCompute(gray2, None)

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    # Using FLANN matcher to match points
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matchesij = matcher.knnMatch(desi, desj, k=2)

    # Ratio test
    good_matches12 = []
    for m, n in matchesij:
        if m.distance < 0.75 * n.distance:
            good_matches12.append(m)
    
    src_ptsi = np.float32([kpi[m.queryIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
    dst_ptsi = np.float32([kpj[m.trainIdx].pt for m in good_matches12]).reshape(-1, 1, 2)

    return src_ptsi, dst_ptsi

# Load the images
img1 = cv2.imread('Given/image_1.jpg')
img2 = cv2.imread('Given/image_2.jpg')
img3 = cv2.imread('Given/image_3.jpg')
img4 = cv2.imread('Given/image_4.jpg')

src_pts1, dst_pts1 = feature_matching(img1, img2)
H12 = compute_homography(src_pts1, dst_pts1)

src_pts2, dst_pts2 = feature_matching(img2, img3)
H23 = compute_homography(src_pts2, dst_pts2)

src_pts3, dst_pts3 = feature_matching(img3, img4)
H34 = compute_homography(src_pts3, dst_pts3)

# Compute cumulative homographies
H13 = np.dot(H23, H12)
H14 = np.dot(H34, H13)

stitched_width = img1.shape[1] + img2.shape[1] + img3.shape[1] + img4.shape[1]
stitched_height = max(img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0])

# Create a large enough canvas to hold the final stitched image
stitched = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

# Place the first image on the canvas
stitched[0:img1.shape[0], 0:img1.shape[1], :] = img1

# Warp the second image onto the canvas
stitched = cv2.warpPerspective(img2, H12, (stitched_width, stitched_height), dst=stitched, borderMode=cv2.BORDER_TRANSPARENT)

# Warp the third image onto the canvas
stitched = cv2.warpPerspective(img3, H13, (stitched_width, stitched_height), dst=stitched, borderMode=cv2.BORDER_TRANSPARENT)

# Warp the fourth image onto the canvas
stitched = cv2.warpPerspective(img4, H14, (stitched_width, stitched_height), dst=stitched, borderMode=cv2.BORDER_TRANSPARENT)

stitched = rescale(stitched, scale=0.25)  # Adjust the scaling factor as needed

cv2.imwrite("results/panorama_screenshot.jpg", stitched)
cv2.waitKey(0)
cv2.destroyAllWindows()
