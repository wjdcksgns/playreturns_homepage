import cv2
import pandas as pd
from glob import glob

def save_function(save_img_path, num, src, label_list, save_name):
    
    cv2.imwrite(save_img_path + f'/{save_name}_{num}.jpg', src)
    
    with open(save_img_path + f'/{save_name}_{num}.txt', "w") as file:
        
        for i in range(len(label_list)):
            
            shape_type, x, y, w, h = label_list[i]

            file.write("%d %.6f %.6f %.6f %.6f\n" % (shape_type, x, y, w, h))
            
def save_num_check():
    
    train_output = len(glob('./dataUtil/model_dataset/model_learn_images/train/*.jpg'))
    val_output = len(glob('./dataUtil/model_dataset/model_learn_images/valid/*.jpg'))
    
    return train_output, val_output