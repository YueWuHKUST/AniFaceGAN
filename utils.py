import numpy as np 
from PIL import Image 
import os 

def save_landmark2D(img, landmark, output_dir, train_step, color='r', step=1):
    """ 
    Parameters:
        img              -- numpy.array, (B, H, W, 3), RGB order, range (0, 255)
        landmark         -- numpy.array, (68, 2), y direction is opposite to v direction
        color            -- str, 'r' or 'b' (red or blue)
    """
    if color =='r':
        c = np.array([255., 0, 0])
    else:
        c = np.array([0, 0, 255.])

    _, H, W = img.shape
    img = (img + 1.0)/2.0 * 255.0
    #landmark[..., 1] = H - 1 - landmark[..., 1]
    landmark = np.round(landmark).astype(np.int32)
    for i in range(landmark.shape[0]):
        # landmark i 
        x, y = landmark[i, 0], landmark[i, 1]
        for j in range(-step, step):
            for k in range(-step, step):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                img[:, v, u] = c
    img = np.transpose(img, [1,2,0])
    Image.fromarray(img.astype(np.uint8)).save(os.path.join(output_dir, "%06d_landmark2D.png"%train_step))