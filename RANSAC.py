import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_video_to_numpy(file_path):
    video_capture = cv2.VideoCapture(file_path)
    
    if not video_capture.isOpened():
        print("Error opening video file.")
        return None

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an empty NumPy array to store the frames
    video_np_array = np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

    frame_index = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            # End of video
            break

        # Store the frame in the NumPy array
        video_np_array[frame_index] = frame

        frame_index += 1

    # Release the video capture
    video_capture.release()

    return video_np_array

# Replace "path/to/your/file.mov" with the actual file path
file_path = r"C:\Users\zhong\OneDrive\Desktop\Foler\IMG_5251.MOV"
video_np_array = read_video_to_numpy(file_path)

if video_np_array is not None:
    print("Successfully converted video to NumPy array.")
    print("Shape of the NumPy array:", video_np_array.shape)
else:
    print("Failed to convert video to NumPy array.")

#RANSAC-based Essential Matrix Decomposition
def estimate_camera_motion_essential(video_np_array):
    camera_motion = []

    # Convert the first frame to grayscale for feature tracking
    prev_frame_gray = cv2.cvtColor(video_np_array[0], cv2.COLOR_BGR2GRAY)

    # Get the number of frames
    num_frames = len(video_np_array)

    for i in range(1, num_frames):
        curr_frame_gray = cv2.cvtColor(video_np_array[i], cv2.COLOR_BGR2GRAY)

        # Feature detection and matching
        orb = cv2.ORB_create()
        prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame_gray, None)
        curr_keypoints, curr_descriptors = orb.detectAndCompute(curr_frame_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_descriptors, curr_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        prev_matched_pts = np.float32([prev_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        curr_matched_pts = np.float32([curr_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        # Define your camera's intrinsic parameters and distortion coefficients
        fx = 1920   # Focal length in x-direction
        fy = 1080  # Focal length in y-direction
        cx = 1920 / 2  # Principal point x-coordinate
        cy = 1080 / 2  # Principal point y-coordinate
        dist_coeff = np.array([...])  # Distortion coefficients

        # Intrinsic matrix
        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]])
        
        # Estimate the fundamental matrix
        fundamental_matrix, mask = cv2.findFundamentalMat(prev_matched_pts, curr_matched_pts, cv2.FM_RANSAC)

        # Extract camera motion from the essential matrix
        essential_matrix = np.dot(np.dot(intrinsic_matrix.T, fundamental_matrix), intrinsic_matrix)
        _, R, t, _ = cv2.recoverPose(essential_matrix, prev_matched_pts, curr_matched_pts, intrinsic_matrix)

        # Append the camera motion estimates to the list
        dx = t[0]
        dy = t[1]
        camera_motion.append((dx, dy))

        # Update the previous frame for the next iteration
        prev_frame_gray = curr_frame_gray

    return camera_motion

# Call the function to estimate camera motion
camera_motion = estimate_camera_motion_essential(video_np_array)

# Calculate the cumulative motion for trajectory plot
trajectory = np.cumsum(np.array(camera_motion), axis=0)

# Plot the trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory RANSAC')
ax = plt.gca()
ax.axis("equal")
plt.xlabel('x [? units]')
plt.ylabel('y [? units]')
plt.title('Trajectory')
plt.legend()
plt.show()
