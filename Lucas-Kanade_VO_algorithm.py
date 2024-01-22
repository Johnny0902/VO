import cv2
import numpy as np

def estimate_camera_motion(video_np_array):
    # Create an empty list to store the camera motion estimates
    camera_motion = []

    # Convert the first frame to grayscale for feature tracking
    prev_frame_gray = cv2.cvtColor(video_np_array[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(video_np_array)):
        # Convert the current frame to grayscale
        curr_frame_gray = cv2.cvtColor(video_np_array[i], cv2.COLOR_BGR2GRAY)

        # Detect feature points using the Shi-Tomasi corner detection
        prev_pts = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

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

        # Update the previous frame for the next iteration
        prev_frame_gray = curr_frame_gray

    return camera_motion

# Call the function to estimate camera motion
camera_motion_estimates = estimate_camera_motion(video_np_array)

# Print the camera motion estimates for each frame
print("Camera Motion Estimates:")
for i, (dx, dy) in enumerate(camera_motion_estimates):
    print(f"Frame {i+1}: dx = {dx}, dy = {dy}")
