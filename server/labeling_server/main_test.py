from glob import glob
from serverUtil.config import Config as cfg

import os
import cv2
import sys
from dataUtil.agumentUtil import agumentutil
from trackerUtil.trackerFuntion import trackerfuntion


# ##TODO: 웹에서 동영상 받기 및 Train or validation 선택

# # Train setting
# trainPath = "./dataUtil/tracker_data/train_tracker/test1.mp4"
# traindataPath = "./dataUtil/model_dataset/model_learn_images/train"

# # val setting
# valPath = './dataUtil/tracker_data/validation_tracker/test2.mp4'
# valdataPath = "./dataUtil/model_dataset/model_learn_images/valid"

# """-------------------------------------------------------------------"""

# video_list = [trainPath, valPath]
# save_path_list = [traindataPath, valdataPath]

# capture_list = []

# tracker_funtion = trackerfuntion()

# for i in range(len(video_list)):
    
#     path = video_list[i]
    
#     # Create a video capture object to read videos
#     cap = cv2.VideoCapture(path)
    
#     # check w, h
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
#     # read first frame
#     success, src = cap.read()
    
#     capture_info = [cap, src]
    
#     capture_list.append(capture_info)
    
#     # quit if unable to read the video file
#     if not success:
#         print('Failed to read video')
#         sys.exit(1)
    
# """-------------------------------------------------------------------"""

# # Processing each object's bouncing box

# tracker_funtion.selectObject(capture_list)

# """-------------------------------------------------------------------"""

# # Determining object names for each bounding box
# # create multitracker
# tracker_funtion.objectNaming()

# """-------------------------------------------------------------------"""

# tracker_funtion.createData(capture_list, width, height)

# """-------------------------------------------------------------------"""

# # 데이터 증식
# agumentutil()

# train_test
train_img_list = glob(cfg.train_img_list)
valid_img_list = glob(cfg.valid_img_list)

print(f'Train Data: {len(train_img_list)}, Test Data: {len(valid_img_list)}')

# txt 파일에 write
with open(cfg.train_txt, 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
    
with open(cfg.valid_txt, 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

print('Dataset are generated!')

print('Start training the Model')
        
# os.chdir('D:/Labeling_Server')
os.system(f"python ./serverUtil/yolov5/train.py \
            --img {cfg.input_size} \
            --batch {cfg.batch_size} \
            --epochs {cfg.epochs} \
            --data {cfg.data_yaml} \
            --weights {cfg.weight_pt} \
            --name {cfg.train_name} \
            --cfg {cfg.model_config}")