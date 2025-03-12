import numpy as np
import cv2
import cmapy
from jaxtyping import Array, Float
from einops import repeat
def save_to_video_mono(data: Float[Array, "N X Y"], filename, fps=30, duration=10, SCALE_UP=4, cmap='viridis'):
    def upscale(x):
        return repeat(x,"X Y C -> (X repeatX) (Y repeatY) C",repeatX=SCALE_UP,repeatY=SCALE_UP)
    def to_uint8(x):
        return (255*(x+1)/2).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #print(data.shape)
    data = np.array(data)
    height,width = data[0].shape
    step_ratio = int(duration*fps/data.shape[0])
    out = cv2.VideoWriter(filename, fourcc, fps, (width*SCALE_UP, height*SCALE_UP),True)
    for i in range(data.shape[0]*step_ratio):
        #frame = np.uint8(cmapy.colorize(frame, cmap=cmap)*255)
        frame = data[i//step_ratio]
        #print(frame.shape)
        frame = cv2.applyColorMap(to_uint8(frame), cmapy.cmap(cmap))
        #print(frame.shape)
        frame = upscale(frame)
        #print(frame.shape)
        out.write(frame)
    out.release()
    return



def save_to_video_rgb(data: Float[Array, "N C X Y"], filename, fps=30, duration=10, SCALE_UP=4, cmap='viridis'):
    def upscale(x):
        return repeat(x,"X Y C -> (X repeatX) (Y repeatY) C",repeatX=SCALE_UP,repeatY=SCALE_UP)
    def to_uint8(x):
        return (255*(x+1)/2).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(data.shape)
    data = np.array(data)
    #height,width = data[0].shape
    height = data.shape[1]
    width = data.shape[2]
    print(height,width)
    step_ratio = int(duration*fps/data.shape[0])
    out = cv2.VideoWriter(filename, fourcc, fps, (width*SCALE_UP, height*SCALE_UP),True)
    for i in range(data.shape[0]*step_ratio):
        #frame = np.uint8(cmapy.colorize(frame, cmap=cmap)*255)
        frame = data[i//step_ratio]
        print(frame.shape)
        frame = to_uint8(frame)
        frame = upscale(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(frame_bgr.shape)
        #print(frame.shape)
        out.write(frame_bgr)
    out.release()
    return

