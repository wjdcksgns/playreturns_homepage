import cv2
import os
from glob import glob
from random import sample, randint, choice
from labeling_server.dataUtil.dataAgument import dataagument

def agumentutil(dir_name, progressbar):
    
    agument = dataagument()

    # 생성된 데이터 Path 불러오기
    img_output = glob(dir_name + '\\train\\' + '*.jpg')
    txt_output = glob(dir_name + '\\train\\' + '*.txt')
    

    # 생성된 데이터대로 반복
    for data_num in range(len(img_output)):
        
        # input img
        get_img = img_output[data_num]
        print(get_img)
        
        img_color = cv2.imread(get_img, cv2.IMREAD_COLOR)
        # img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # input txt
        get_txt = txt_output[data_num]
        label_info = []              
        with open(get_txt, "r") as f:
            for line in f:
                label_info.append(line.strip().split(' '))
                
        imgfilename = os.path.basename(get_img)
        txtfilename = os.path.basename(get_txt)
        
        if img_color is not None:
            
            # img agument
            agument.input_img(img_color, label_info)

            # print('trans_bg')
            agument.agumentActivate(None, imgfilename, txtfilename, 'trans_bg', dir_name, 5) # defult = 10
            
            # print('rotate')
            agument.agumentActivate(agument.rotate_img(), imgfilename, txtfilename, 'rotate', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))
        
            # print('crop')
            agument.agumentActivate(agument.crop_img(), imgfilename, txtfilename, 'crop', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))
            
            # print('cutout')
            agument.agumentActivate(agument.cutout_img(), imgfilename, txtfilename, 'cutout', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))
            
            # print('trans_rgb')
            agument.agumentActivate(agument.trans_rbg_img(), imgfilename, txtfilename, 'trans_rgb', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))

            # print('brightness')
            agument.agumentActivate(agument.brightness_img(), imgfilename, txtfilename, 'brightness', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))
            
            # print('blur')
            agument.agumentActivate(agument.blur_img(), imgfilename, txtfilename, 'blur', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))
            
            # print('noise')
            agument.agumentActivate(agument.noise_img(), imgfilename, txtfilename, 'noise', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))

            # print('flip)
            agument.agumentActivate(agument.flip_img(), imgfilename, txtfilename, 'flip', dir_name, [choice([True, False]) for v in range(1, 4)].count(True))


        progressbar.progress_cnt()
        
    

    # Total number of images
    img_num = int(len(img_output)) - 1
    
    # number of images to select
    choice_num = int(img_num * 0.3) 
    
    # index of randomly selected image
    choice_path = sample(range(0, img_num), choice_num) 

    for ch_path in choice_path:
        
        img_path = img_output[ch_path]
        txt_path = txt_output[ch_path]
        
        save_file = f'{dir_name}/train/GRAYIMG_{randint(1, 50000)}{randint(1, 50000)}'
        
        try:
            imgtogray = img_color = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'{save_file}.jpg', imgtogray)
            
            txt_info = []              
            with open(txt_path, "r") as f:
                for line in f:
                    txt_info.append(line.strip().split(' '))

            # breakpoint()

            with open(f'{save_file}.txt', "w") as file:
                
                for i in range(len(txt_info)):
                    
                    shape_type, x, y, w, h = txt_info[i]

                    file.write("%d %.6f %.6f %.6f %.6f\n" % (int(shape_type), float(x), float(y), float(w), float(h)))
            
            
        except Exception as e:
            print(e)
            pass
        
        progressbar.progress_cnt()