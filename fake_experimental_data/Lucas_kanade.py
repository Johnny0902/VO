import cv2
import numpy as np
from os.path import join,exists
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
file_path = r"C:\Users\zhong\OneDrive\Desktop\Foler\IMG_5251.mov"
file_path = join('C:\\', "Users", "zhong", "OneDrive", "Desktop", "Foler", "IMG_5251.mov")
video_np_array = read_video_to_numpy(file_path)

if video_np_array is not None:
    print("Successfully converted video to NumPy array.")
    print("Shape of the NumPy array:", video_np_array.shape)
else:
    print("Failed to convert video to NumPy array.")

#Lucan-Kanade_VO_algorithm
def estimate_camera_motion(video_np_array):
    # Create an empty list to store the camera motion estimates
    camera_motion = []

    # Convert the first frame to grayscale for feature tracking
    prev_frame_gray = cv2.cvtColor(video_np_array[0], cv2.COLOR_BGR2GRAY)

    # Get the number of frames
    num_frames = len(video_np_array)

    # Define numpy arrays to store dx and dy values for each frame
    dxs = np.zeros((num_frames, 1))
    dys = np.zeros((num_frames, 1))

    for i in range(1, num_frames):
        # Convert the current frame to grayscale
        curr_frame_gray = cv2.cvtColor(video_np_array[i], cv2.COLOR_BGR2GRAY)

        # Detect feature points using the Shi-Tomasi corner detection
        #prev_pts = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        prev_pts = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=1)

        # Calculate optical flow using Lucas-Kanade method
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_pts, None)

        # Select only the valid points that were successfully tracked
        prev_pts = prev_pts[status == 1]
        curr_pts = curr_pts[status == 1]

        # Calculate the affine transformation between the points in the two frames
        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        # The camera motion can be extracted from the affine transformation matrix
        # The translation (dx, dy) represents the camera motion in each frame
        dx = transform[0, 2]
        dy = transform[1, 2]

        # Append the camera motion estimates to the list
        camera_motion.append((dx, dy))

        # Update the dxs and dys arrays
        dxs[i] = dx
        dys[i] = dy

        # Update the previous frame for the next iteration
        prev_frame_gray = curr_frame_gray

    return dxs, dys

if __name__ == '__main__':
    # Call the function to estimate camera motion
    dxs, dys = estimate_camera_motion(video_np_array)

    # Print the camera motion estimates for each frame
    print("Camera Motion Estimates:")
    for i, (dx, dy) in enumerate(zip(dxs, dys)):
        print(f"Frame {i+1}: dx = {dx[0]}, dy = {dy[0]}")

    # assumes dxs and dys are a collection of estimated distance travelled per frame
    plt.plot(np.cumsum(dxs), np.cumsum(dys), label='Trajectory Lucas Kanade')
    ax = plt.gca()
    ax.axis("equal")
    plt.xlabel('x [? units]')
    plt.ylabel('y [? units]')
    plt.title('Trajectory')
    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    plt.legend()
    plt.show()