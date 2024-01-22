import cv2
import numpy as np
from import_5251 import import5251
from matplotlib import pyplot as plt


if __name__ == '__main__':
    video_np_array = import5251()
    fake_np_array = np.zeros((106, 640, 420, 3))

    for i in range (53):
        fake_np_array[i,:,:,:] = video_np_array[0,i:640+i,i:420+i,:]
    for ii, i in enumerate(range (52,0,-1)):
        fake_np_array[53+ii,:,:,:] = video_np_array[0,i:640+i,i:420+i,:]

    size = 640, 420
    duration = 106/30  
    fps = 30  
    out = cv2.VideoWriter('fake_diagonal.mov', cv2.VideoWriter_fourcc(*'XVID'), fps, (size[1], size[0]))  
    for i in range(106):  
        data = fake_np_array[i,:,:,:].astype(np.uint8)
        out.write(data)  
    out.release() 