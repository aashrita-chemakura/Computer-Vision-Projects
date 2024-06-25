#Importing the required libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

# (1.1) - Detecting and plotting the pixel coordinates

# Read the Video 
vid = cv2.VideoCapture('given/ball.mov') 
centre_X = []
centre_Y = []

while(vid.isOpened()):

    ret, frame = vid.read()
    if ret == True:
        vid_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the red color in HSV
        lower_r = np.array([2, 175, 120])
        higher_r = np.array([179, 232, 253])

    # Create an HSV mask to extract the red color
        vid_mask = cv2.inRange(vid_hsv, lower_r, higher_r)

    # Convert into a binary image 
        ret, vid_bi = cv2.threshold(vid_mask,127,255, cv2.THRESH_BINARY)

    # Calculating moments of the binary image
        Moments = cv2.moments(vid_bi)
 
    # Calculating the coordinates of center
        try:
            X = float(Moments["m10"] /Moments["m00"])
            Y = float(Moments["m01"] / Moments["m00"])
            np.array(centre_X.append(X))
            np.array(centre_Y.append(Y))

        except ZeroDivisionError:
            pass

        for a in range(len(centre_X)):
            cv2.circle(frame, (int(centre_X[a]), int(centre_Y[a])),2,(255,255,255), -1)
    
        cv2.imshow('frame', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    else:
        break

plt.scatter(centre_X, -np.array(centre_Y))

# (1.2) - Using of Lease Squares to fit a curve 
centre_X = np.array(centre_X).T
centre_X = centre_X.reshape([len(centre_X),1])
centre_Y = np.array(centre_Y).T
centre_Y = centre_Y.reshape([len(centre_Y),1])
D = np.hstack([centre_X**2,centre_X,np.ones([len(centre_X),1])])
P = np.linalg.inv(np.dot(D.T,D))
Q = np.dot(D.T,centre_Y)
B = np.dot(P,Q)

plt.plot(centre_X,-(B[0]*(centre_X**2))-(B[1]*centre_X)-B[2], 'r')
plt.show()

# (1.3) - Considering origin to be at top-left
a = B[0]
b = B[1]
c = B[2]
new_Y = 300 + centre_Y[0]
c -= new_Y 

s_1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
s_2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

print("\nEquation of curve is", a, "x**2+", b, "x+", c)
print("The x- coordinate of the landing spot of the ball is-", max(s_1, s_2),"\n")

vid.release()
cv2.destroyAllWindows()
