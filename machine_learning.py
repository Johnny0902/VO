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
file_path = r"C:\Users\zhong\OneDrive\Desktop\Foler\fake_diagonal.MOV"
video_np_array = read_video_to_numpy(file_path)

if video_np_array is not None:
    print("Successfully converted video to NumPy array.")
    print("Shape of the NumPy array:", video_np_array.shape)
else:
    print("Failed to convert video to NumPy array.")
    
#Machine learning algorithm
def calculate_camera_motion(prev_frame, curr_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors using ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    # Match features using Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calculate the essential matrix
    E, _ = cv2.findEssentialMat(src_pts, dst_pts)
    
    # Recover camera motion from essential matrix
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
    
    return R, t
x_trajectory = np.zeros((len(video_np_array)))
y_trajectory = np.zeros((len(video_np_array)))
z_trajectory = np.zeros((len(video_np_array)))

# Iterate through video frames and estimate camera motion
for i in range(1, len(video_np_array)-1):
    prev_frame = video_np_array[i - 1]
    curr_frame = video_np_array[i]
    
    R, t = calculate_camera_motion(prev_frame, curr_frame)
    """
    try:
        R, t = calculate_camera_motion(prev_frame, curr_frame)
    except:
        print('Error!')
    """
    
    # Now you have rotation matrix R and translation vector t
    # You can integrate these for trajectory estimation
    x_trajectory[i] = x_trajectory[i-1] + t[0]
    y_trajectory[i] = y_trajectory[i-1] + t[1]
    z_trajectory[i] = z_trajectory[i-1] + t[2]



    #print(R)
    print(t)
    # Print or visualize results as needed
    #print("Frame:", i)
    #print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)

# assumes dxs and dys are a collection of estimated distance travelled per frame
plt.plot(x_trajectory,y_trajectory)
ax = plt.gca()
ax.axis("equal")
plt.xlabel('x [? units]')
plt.ylabel('y [? units]')
plt.title('Trajectory')
plt.legend()
plt.savefig("up_down_ML.png")
plt.show()