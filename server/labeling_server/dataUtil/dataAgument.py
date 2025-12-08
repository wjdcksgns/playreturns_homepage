from random import randint, random, uniform, choice
from glob import glob

import numpy as np
import cv2
import copy


def yoloformat(corner, H, W):
    center_bbox_x = corner[1] + (corner[3] / 2)
    center_bbox_y = corner[2] + (corner[4] / 2)

    return corner[0], round(center_bbox_x / W, 7), round(center_bbox_y / H, 7), round(corner[3] / W, 7), round(corner[4] / H, 7)


class dataagument():
    
    def __init__(self) -> None:
        pass
    
    
    def input_img(self, img, label_list):
        '''
        self.img : 초기 이미지값
        self.h,w : 초기 이미지의 높이, 너비값
            - 3차원(RGB) 데이터만 입력받게끔 설정
        self.label : 이미지값에 대한 좌표값으로, centerx, y, w, h 비율값으로 저장됨
        '''
        
        self.img = img
        
        self.h, self.w = self.img.shape[:2]

        self.label_list = label_list


    def adapt_bbox(self, label):
        '''
        h, w : label의 값은 비율값, 즉 1 미만의 float 데이터이므로 h, w를 입력받는다
            - 입력받은 값을 곱해주면 기존의 bbox 좌표 위치가 명확해진다.
        cv2.rectangle의 위치는 좌하단이 아닌 좌상단에서 시작되므로, 좌상단의 x,y 좌표를
        구하는 것이 이 함수의 목적이다.
        '''
        x, y = float(label[1]) * self.w, float(label[2]) * self.h
        w, h= float(label[3]) * self.w, float(label[4]) * self.h

        x -= (w / 2)
        y -= (h / 2)

        bbox_x, bbox_y, bbox_h, bbox_w = int(x), int(y), int(h), int(w)

        return label[0], bbox_x, bbox_y, bbox_h, bbox_w


    def brightness_img(self):
        '''
        1. [brightness]

        입력받은 이미지의 밝기를 조절하는 agument
        val : 밝기를 조절하는 변수로, 0 ~ 150 정도가 좋을 듯함.
        type : True이면 밝게, False면 어둡게 val값만큼 조정한다.
        '''
        
        bn_img = copy.deepcopy(self.img)
        
        val = randint(50, 151)
        
        # 랜덤으로 밝기 조절
        array = np.full(bn_img.shape, val, dtype=np.uint8)
        
        # 랜덤으로 T/F 선택
        type = choice([True, False])
        
        if type == True:

            bn_img = cv2.add(bn_img, array)  # 배열 더하기
        
        else:

            bn_img = cv2.subtract(bn_img, array) # 배열 빼기
        
        return bn_img, self.label_list


    def blur_img(self):
        '''
        2. [blur]

        입력받은 이미지를 흐릿하게 조절하는 agument
        가중치를 픽셀에 적용하는 Gaussianblur 필터를 사용할 예정

        cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
            - src : 입력 이미지
            - ksize : 가우시안 커널 크기, (0,0) 지정시 sigma 값에 의해 자동 결정 
                        / 0을 제외하면 홀수값만 가능
            - sigmaX : x방향 sigma(표준편차) => 입력받을 변수로 설정
            - sigmaY : y방향 sigma, 0이면 sigmaX와 동일하게 변경
            - borderType : 가장자리 픽셀 확장 방식

        따라서 simgax 값만 사용자 정의로 받으며 1 ~ 5 정도가 좋을 듯함.
        '''
        
        b_img = copy.deepcopy(self.img)
        
        # 랜덤으로 흐림도 조절
        sigma = int(randint(1, 5))
        
        if sigma < 1:

            raise Exception("sigma should be 'int' and bigger than '0'")

        else:

            b_img = cv2.GaussianBlur(b_img, (0, 0), sigma)

            return b_img, self.label_list


    def noise_img(self):
        '''
        3. [noise]

        입력받은 이미지에 노이즈를 추가하는 agument
        가장 대중적인 'SaltPepper' noise 코드를 사용
        per : noise를 설정하는 변수로, 높을수록 noise가 심해진다.
            - 단, 0~1 사이의 값만 받는다.
        '''
        
        n_img = copy.deepcopy(self.img)
        
        per = float(random())
        
        if per > 1:
            raise Exception("per should be smaller than 1")
            
        result = copy.deepcopy(n_img)
        number_of_pixels = int((self.h * self.w / 10) * per)

        for i in range(number_of_pixels):
            y_coord = randint(0, self.h-1)
            x_coord = randint(0, self.w-1)

            if result.ndim > 2:

                result[y_coord][x_coord] = [randint(0,255), randint(0,255), randint(0,255)]
            
            else:

                result[y_coord][x_coord] = 255
            
        for i in range(number_of_pixels):
            y_coord = randint(0, self.h-1)
            x_coord = randint(0, self.w-1)

            if result.ndim > 2:

                result[y_coord][x_coord] = [randint(0,255), randint(0,255), randint(0,255)]
            
            else:

                result[y_coord][x_coord] = 0

        n_img = result
        
        return n_img, self.label_list


    def cutout_img(self):
        '''
        4. [cutout]

        입력받은 이미지의 일부를 흑백처리하여 공백처럼 보이게 하는 agument
        min_ksize : 최소 커널 사이즈
        max_ksize : 최대 커널 사이즈
        cnt : cutout box의 개수
        '''
        
        co_img = copy.deepcopy(self.img)
        
        # 랜덤으로 흑백처리 사이즈 선택
        ksize = randint(5, 101)
        
        # 랜덤으로 반복 횟수 선택
        cnt = randint(1, 6)
        
        for i in range(cnt):

            start_x = randint(0, self.w - ksize)
            start_y = randint(0, self.h - ksize)

            co_img[start_x:start_x+ksize, start_y:start_y+ksize] = [0, 0, 0]
        
        return co_img, self.label_list


    def crop_img(self):
        '''
        5. [crop]

        입력받은 이미지를 zoom-in 하여 특정 부분만 확대하는 agument
        bbox 좌표의 변화가 발생하므로, 좌표값에 대한 추가 작업 실시

        rate : 얼만큼 zoom-in 할 것인지에 대한 변수로, 0~0.35 사이가 좋을 듯함.
        - 0.5 이상부터 안되지만 0.49이어도 많이 확대되므로 데이터셋에 좋지 않을 것 같음.
        '''
        # rate = float(uniform(0.1, 0.5))
        rate = float(uniform(0.1, 0.36))
        
        # rate 만큼 shape의 h,w 값을 활용해서 새로운 x,y 좌표를 구한다.
        start_x = int(self.w * rate)
        end_x = int(self.w - start_x)
        start_y = int(self.h * rate)
        end_y = int(self.h - start_y)

        while True:
            
            global cr_img
            
            try:
                # self.img의 중앙을 기준으로 양 옆, 위아래를 rate 길이만큼 깎는다.
                cr_img = copy.deepcopy(self.img[start_y:end_y, start_x:end_x])
                # cr_img = self.img[start_x:end_x, start_y:end_y].copy()
               
                # 깎인 이미지로 인해 조정된 shape를 다시 구한다.
                nsize_h, nsize_w, _ = cr_img.shape
                    
                # 조정된 img를 다시 기존 shape로 재조정한다.
                cr_img = cv2.resize(cr_img, dsize = (self.w, self.h), interpolation=cv2.INTER_AREA)
                
            except Exception as e:
                
                print("예외 상황 발생 : ", e)

            if cr_img is not None:
                
                #print("crop 성공 및 resize 성공")
                
                break
            
        label_info = []
        for label_num in range(len(self.label_list)):
            
            label = self.label_list[label_num]

            # breakpoint()
            
            # 기존의 bbox 좌표를 구한다.
            cls, nx, ny, nh, nw = self.adapt_bbox(label)

            # 기존 bbox에서 rate만큼 좌표값을 깎는다.
            start_nx = int(self.w * rate)
            start_ny = int(self.h * rate)

            nx -= start_nx
            ny -= start_ny
            
            try:
                # (처음 조정된 shape / 재조정된 shape) 만큼 bbox 좌표 비율을 늘린다.
                bbox_x = int((nx * (self.w / nsize_w)))
                bbox_y = int((ny * (self.h / nsize_h)))
                bbox_h = int((nh * (self.h / nsize_h)))
                bbox_w = int((nw * (self.w / nsize_w)))

                if (bbox_x + bbox_w < 0) or (bbox_y + bbox_h < 0):

                    pass

                else:
                
                    if bbox_x <= 0 :   # x가 음수라면,
                        bbox_w -= abs(bbox_x) # 너비에서 절대값 x를 빼준다.
                        bbox_x = 0
                        
                    if bbox_x >= self.w: # x가 w 이상 이라면,
                        bbox_w = bbox_x - bbox_w # x는 w보다 크므로 x에서 w를 빼준다.
                        bbox_x = self.w
                        
                    if bbox_y <= 0 :
                        bbox_h -= abs(bbox_y)
                        bbox_y = 0
                        
                    if bbox_y >= self.h:
                        bbox_h = bbox_y - bbox_h
                        bbox_y = self.h
                        
                    if bbox_x + bbox_w > self.w:
                        bbox_w = self.w - bbox_x
                        
                    if bbox_y + bbox_h > self.h:
                        bbox_h = self.h - bbox_y

                    bbox_label = [cls, bbox_x, bbox_y, bbox_w, bbox_h]
                    
                    label_info.append(yoloformat(bbox_label, self.h, self.w))
               
            except Exception as e:

                print('Error: ', e)
                pass

        return cr_img, label_info

    
    def trans_rbg_img(self):
        '''
        6. [trans_rbg]

        입력받은 이미지 내 배경색의 red, green, blue를 부각시키는 agument

        rgb : 배경색의 색상을 정하는 변수로, 1이면 blue, 2이면 green, 3이면 red 색상으로 변한다.
        value : 얼만큼 trans 할 것인지에 대한 변수로, 50~150 사이가 좋을 듯함.
        '''
        
        
        # 랜덤으로 r, g, b 선택
        rgb = randint(1, 3)

        img_rgb = copy.deepcopy(self.img)
        
        # change color
        b, g, r = cv2.split(img_rgb)
        zeros = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype='uint8')

        if rgb == 1:
            img_rgb = cv2.merge([b, zeros, zeros])

        elif rgb == 2:
            img_rgb = cv2.merge([zeros, g, zeros])

        elif rgb == 3:
            img_rgb = cv2.merge([zeros, zeros, r])

        else:
            raise Exception('rgb should be 1 or 2 or 3')

        # input crop img
        for lb in self.label_list:
            cls, x, y, h, w = self.adapt_bbox(lb)
        
            crop_img = self.img[y:y+h, x:x+w]

            img_rgb[y:y+h, x:x+w] = crop_img
            
        return img_rgb, self.label_list


    def bg_import(self):
        
        # 기존 이미지
        origin_img = copy.deepcopy(self.img)

        # 배경 이미지
        bg_img_list = sorted(glob('./labeling_server/dataUtil/background/*.jpg'))
        # print('bg_img_list length:', len(bg_img_list))

        # 배경 이미지 불러오기
        bg_img = cv2.resize(cv2.imread(bg_img_list[randint(0, 49)], cv2.IMREAD_COLOR), (origin_img.shape[1], origin_img.shape[0]))
        
        return bg_img, origin_img
    

    def trans_bg_img(self, bg_img, origin_img):
        '''
        7. [trans_bg]

        입력받은 이미지 내 배경을 임의로 선정한 배경 이미지 50가지 중 하나로 변경하는 agument
        '''
        # input crop img
        for lb in self.label_list:
            cls, x, y, h, w = self.adapt_bbox(lb)
        
            crop_img = origin_img[y:y+h, x:x+w]

            bg_img[y:y+h, x:x+w] = crop_img
            
        return bg_img, self.label_list

    
    def rotate_img(self):
        '''
        8. [rotate]

        입력받은 이미지를 90, 180 두 가지 각도로 rotate하는 agument
        bbox 좌표의 변화가 발생하므로, 좌표값에 대한 추가 작업 실시

        angle : 어느 각도로 rotate 할 것인지에 대한 변수로, 90 or 180 이다. 

        *** 
        기존 agument의 rectangle용 bbox는 (좌상단x, 좌상단y, h, w)
        rotate의 bbox는 (좌상단x, 좌상단y, 우하단x, 우하단y) 값이 나온다.
        ***
        '''
        
        r_img = copy.deepcopy(self.img)
        
        angle = choice([int(90), int(180)])
        
        def yoloFormattocv(x1, y1, x2, y2, H, W):
            bbox_width = x2 * W
            bbox_height = y2 * H
            center_x = x1 * W
            center_y = y1 * H

            voc = []

            voc.append(center_x - (bbox_width / 2))
            voc.append(center_y - (bbox_height / 2))
            voc.append(center_x + (bbox_width / 2))
            voc.append(center_y + (bbox_height / 2))

            return [int(v) for v in voc]


        # def cvFormattoYolo(corner, H, W):
        #     bbox_W = corner[3] - corner[1]
        #     bbox_H = corner[4] - corner[2]

        #     center_bbox_x = (corner[1] + corner[3]) / 2
        #     center_bbox_y = (corner[2] + corner[4]) / 2

        #     return corner[0], round(center_bbox_x / W, 5), round(center_bbox_y / H, 5), round(bbox_W / W, 5), round(bbox_H / H, 5)


        height, width = r_img.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_angle = angle * np.pi / 180
        rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])


        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        rotated_mat = cv2.warpAffine(r_img, rotation_mat, (bound_w, bound_h))

        new_height, new_width = rotated_mat.shape[:2]
        H, W = r_img.shape[:2]

        label_info = []
        for label_num in range(len(self.label_list)):
            
            box_label = self.label_list[label_num]

            (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(
                float(box_label[1]), float(box_label[2]), float(box_label[3]), float(box_label[4]), H, W)

            upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
            upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
            lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
            lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

            new_lower_right_corner = [-1, -1]
            new_upper_left_corner = []        

            for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift, lower_right_corner_shift):
                new_coords = np.matmul(rot_matrix, np.array((i[0], -i[1])))
                x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                if new_lower_right_corner[0] < x_prime:
                    new_lower_right_corner[0] = x_prime
                if new_lower_right_corner[1] < y_prime:
                    new_lower_right_corner[1] = y_prime

                if len(new_upper_left_corner) > 0:
                    if new_upper_left_corner[0] > x_prime:
                        new_upper_left_corner[0] = x_prime
                    if new_upper_left_corner[1] > y_prime:
                        new_upper_left_corner[1] = y_prime
                else:
                    new_upper_left_corner.append(x_prime)
                    new_upper_left_corner.append(y_prime)

            new_bbox = [box_label[0], new_upper_left_corner[0], new_upper_left_corner[1], new_lower_right_corner[0], new_lower_right_corner[1]]
            nx,ny,xx,xy = int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]), int(new_bbox[4])
            w, h = xx - nx, xy - ny
            cx, cy = int(nx + w/2), int(ny + h/2)

            cx, cy, w, h = cx / new_width, cy / new_height, w / new_width, h / new_height

            label_info.append([new_bbox[0], cx, cy, w, h])
        
        return rotated_mat, label_info
            
    
    def flip_img(self):
        '''
        9. [flip]

        입력받은 이미지를 좌우반전하는 agument
        flip_num : 0이면 상하, 1이면 좌우반전
        '''
        
        img = copy.deepcopy(self.img)

        # 랜덤으로 상하 혹은 좌우 반전
        flip_num = randint(0, 1)
        
        flip_img = cv2.flip(img, flip_num)

        label_info = []

        for label_list in self.label_list:

            label = self.adapt_bbox(label_list)

            cls, x, y, h, w = label

            if flip_num == 0:

                y = self.h - (y + h)

            elif flip_num == 1:

                x = self.w - (x + w)

            cx, cy = x + (w / 2), y + (h / 2)
            cx, cy, w, h = cx / self.w, cy / self.h, w / self.w, h / self.h

            label_info.append([int(cls), cx, cy, w, h])

        return flip_img, label_info


    def agumentActivate(self, function_name, imgfilename, txtfilename, f_type, dir_name, num):
        
        save_path = dir_name + '\\train\\'
        
        for f_iterations in range(num):
            
            if f_type == 'trans_bg':
                bg_img, origin_img = self.bg_import()
                result_img, result_label = self.trans_bg_img(bg_img, origin_img)
            
            else:
                result_img, result_label = function_name
            
            name = imgfilename.strip(".jpg")
            
            try:
                cv2.imwrite(f'{save_path}{name}_{f_type}{f_iterations}.jpg', result_img)
                
                with open(f'{save_path}{name}_{f_type}{f_iterations}.txt', "w") as file:
                    
                    for i in range(len(result_label)):
                        
                        shape_type, x, y, w, h = result_label[i]

                        file.write("%d %.6f %.6f %.6f %.6f\n" % (int(shape_type), float(x), float(y), float(w), float(h)))
                        
            except Exception as e:
                print(e)
                pass