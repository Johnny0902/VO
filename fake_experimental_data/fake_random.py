import cv2
import numpy as np
from import_5251 import import5251
from Lucas_kanade import estimate_camera_motion
from matplotlib import pyplot as plt


if __name__ == '__main__':
    video_np_array = import5251()
    size_h = 640
    size_w = 420
    fake_np_array = np.zeros((106, size_h, size_w, 3))
    pos_x = 1000
    pos_y = 1000
    pos_xs = np.zeros((106,))
    pos_ys = np.zeros((106,))
    vel_x = 2
    vel_y = 2
    max_acceleration = 1

    for i in range (106):
        ddx = np.random.randint(-max_acceleration, max_acceleration)
        ddy = np.random.randint(-max_acceleration, max_acceleration)
        vel_x = vel_x + ddx
        vel_y = vel_y + ddy
        pos_x = pos_x + vel_x
        pos_y = pos_y + vel_y
        pos_x = np.clip(pos_x, 0, 500)
        pos_y = np.clip(pos_y, 0, 500)
        pos_xs[i] = pos_x
        pos_ys[i] = pos_y
        fake_np_array[i,:,:,:] = video_np_array[0,pos_y:pos_y+size_h,pos_x:pos_x+size_w,:]

    size = 640, 420
    duration = 106/30  
    fps = 30  
    out = cv2.VideoWriter('fake_random.mov', cv2.VideoWriter_fourcc(*'XVID'), fps, (size[1], size[0]))  
    for i in range(106):  
        data = fake_np_array[i,:,:,:].astype(np.uint8)
        out.write(data)  
    out.release()

dxs, dys = estimate_camera_motion(video_np_array)

# Print the camera motion estimates for each frame
print("Camera Motion Estimates:")
for i, (dx, dy) in enumerate(zip(dxs, dys)):
    print(f"Frame {i+1}: dx = {dx[0]}, dy = {dy[0]}")

# assumes dxs and dys are a collection of estimated distance travelled per frame
plt.plot(np.cumsum(dxs), np.cumsum(dys), label='Trajectory Lucas Kanade')
plt.plot(pos_xs, pos_ys, label = "true")
ax = plt.gca()
ax.axis("equal")
plt.xlabel('x [? units]')
plt.ylabel('y [? units]')
plt.title('Trajectory')
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
plt.legend()
plt.show()