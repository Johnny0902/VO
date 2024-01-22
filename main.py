import cv2
import numpy as np

# Initialize ORB detector
orb = cv2.ORB_create()

# Initialize brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize camera matrix (replace with your camera's intrinsic parameters)
# The focal length (fx, fy) and principal point (cx, cy) can be obtained from camera calibration
fx = 500  # Focal length in pixels
fy = 500
cx = 320  # Principal point in pixels
cy = 240

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Function to estimate camera motion between two frames
def estimate_camera_motion(prev_img, curr_img):
    # Convert images to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Find key points and descriptors in both frames
    prev_kps, prev_desc = orb.detectAndCompute(prev_gray, None)
    curr_kps, curr_desc = orb.detectAndCompute(curr_gray, None)

    # Match descriptors using the brute-force matcher
    matches = bf.match(prev_desc, curr_desc)

    # Extract matched key points
    prev_matched_kps = np.array([prev_kps[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    curr_matched_kps = np.array([curr_kps[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Compute the essential matrix using RANSAC
    E, _ = cv2.findEssentialMat(curr_matched_kps, prev_matched_kps, K)

    # Recover the camera's rotation and translation from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, curr_matched_kps, prev_matched_kps, K)

    return R, t

# Test the visual odometry algorithm
def main():
    cap = cv2.VideoCapture(0)

    prev_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if prev_frame is not None:
            R, t = estimate_camera_motion(prev_frame, frame)

            # Print the camera's translation and rotation
            print("Translation:", t.ravel())
            print("Rotation:\n", R)

        prev_frame = frame.copy()

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()