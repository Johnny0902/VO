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

def import5251():
    # Replace "path/to/your/file.mov" with the actual file path
    file_path = r"C:\Users\zhong\OneDrive\Desktop\Foler\IMG_5251.MOV"
    video_np_array = read_video_to_numpy(file_path)

    if video_np_array is not None:
        print("Successfully converted video to NumPy array.")
        print("Shape of the NumPy array:", video_np_array.shape)
    else:
        print("Failed to convert video to NumPy array.")

    return video_np_array