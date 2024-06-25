import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R

def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def Hough_Transform(rho, theta, threshold, min_line_length, max_line_gap):
    height, width, _ = frame.shape
    edges = np.argwhere(paper_edges != 0)
    dist_max = np.ceil(np.sqrt(height * height + width * width))
    H_mat = np.zeros((2 * int(dist_max), 180))
    for y, x in edges:
        for t in range(0, 180):
            r = int(round((x * np.cos(t * theta)) + (y * np.sin(t * theta))) + dist_max)
            H_mat[r][t] += 1

    lines = []
    indexes = np.argwhere(H_mat > threshold)
    for r, t in indexes:
        x1 = int(np.cos(t * theta) * (r - dist_max) + max_line_gap * (-(np.sin(t * theta))))
        y1 = int(np.sin(t * theta) * (r - dist_max) + max_line_gap * (np.cos(t * theta)))
        x2 = int(np.cos(t * theta) * (r - dist_max) - max_line_gap * (-(np.sin(t * theta))))
        y2 = int(np.sin(t * theta) * (r - dist_max) - max_line_gap * (np.cos(t * theta)))
        line = ((x1, y1), (x2, y2))
        if line not in lines:
            lines.append(line)
    return lines

def find_intersection(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1 = lines[i][0]
            x2, y2 = lines[i][1]
            x3, y3 = lines[j][0]
            x4, y4 = lines[j][1]

            m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            b1 = y1 - m1 * x1 if x2 != x1 else float('inf')
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
            b2 = y3 - m2 * x3 if x4 != x3 else float('inf')

            if m1 == m2:
                continue

            x = (b2 - b1) / (m1 - m2) if m1 != float('inf') and m2 != float('inf') else 0
            y = m1 * x + b1 if m1 != float('inf') else 0

            if x1 <= x <= x2 or x2 <= x <= x1:
                if y1 <= y <= y2 or y2 <= y <= y1:
                    if x3 <= x <= x4 or x4 <= x <= x3:
                        if y3 <= y <= y4 or y4 <= y <= y3:
                            if paper_edges[int(y), int(x)] == 0:
                                intersections.append((x, y))
    return intersections

def Find_Homography(src, des):
    A = []
    for i in range(len(des)):
        x, y = src[i][0], src[i][1]
        u, v = des[i][0], des[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)

    d, s, v = np.linalg.svd(A)
    Vt = np.transpose(v)

    H = Vt[:, -1]
    H = H / H[-1]
    H_matrix = np.reshape(H, (3, 3))

    return H_matrix

def decompose_homography(H):
    U, s, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        R = np.dot(U, np.dot(np.diag([1, 1, -1]), Vt))
    T = H[:, 2] / s[0]
    return R, T

vid = cv2.VideoCapture('Given/project2.avi')
start_time = time.time()
duration = 10  # Duration in seconds to run the video

while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        frame = rescale(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (19, 19), 0)
        paper_edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        rho = 10
        theta = np.pi / 180
        threshold = 100
        min_line_length = 1000
        max_line_gap = 1000

        lines = Hough_Transform(rho, theta, threshold, min_line_length, max_line_gap)
        top_lines = sorted(lines, key=lambda x: x[0][1])[-20:]
        line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        for line in lines:
            cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

        intersections = find_intersection(lines)
        for x, y in intersections:
            centre = (int(x), int(y))
            cv2.circle(frame, centre, 3, 255, 2)

        if len(intersections) >= 4:
            des = np.array([[216, 0], [216, 279], [0, 279], [0, 0]])
            homography = Find_Homography(intersections[:4], des)
            print(homography)

            R, t = decompose_homography(homography)
            print(R, t)

        cv2.imshow('frame', frame)

    if time.time() - start_time > duration:
        break

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
