import cv2
import pandas as pd

from time import sleep
from labeling_server.trackerUtil.createTracker import createTrackerByName
from labeling_server.saveUtil.saveFunction import save_function, save_num_check
from labeling_server.dataUtil.objectDataCheck import object_insert



class trackerfuntion:

    def __init__(self):
        
        self.all_selectbox = []
        self.all_selectbox_frame = []
        self.multiTracker_list = []
        self.objectnames = []
        self.roi_sets = []
        
        ##TODO: 웹에서 trackerType 받기
        # self.trackerType = input("input trackerType")    # CSRT
        self.trackerType = 'CSRT'
        # self.object_data = object_check()
        
        # SIFT 관련
        self.nndrRatio = 0.75

    def initialize(self):
        self.all_selectbox.clear()
        self.all_selectbox_frame.clear() 
        self.multiTracker_list.clear() 
        self.objectnames.clear() 
        self.roi_sets.clear() 

    ## TODO: CAP별로 프레임 저장 후 언팩
    def selectObject(self, capture_list):
        '''
        입력받은 프레임 값을 selectROI를 통해 사용자가 임의로 bbox를 설정
        키 값을 입력받고 팝업창이 사라지면 현재 프레임에 작성된 모든 bbox, frame 값을 보관한다.
        이는 train, valid 둘 다 동일하게 작업된다.
        '''

        for capture_num in range(len(capture_list)):
            
            # train or valid mp4 파일에 대한 프레임 값
            first_frame = capture_list[capture_num][1]
        
            # Select boxes
            bboxes = []
            
            # draw bounding boxes over objects
            # selectROI's default behaviour is to draw box starting from the center
            # when fromCenter is set to false, you can draw box starting from top left corner
            while True:
                
                bbox = cv2.selectROI('MultiTracker', first_frame)
                bboxes.append(bbox)
                
                
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                
                # q is pressed
                k = cv2.waitKey(0) & 0xFF
                if (k==113):
                    break
                
            cv2.destroyWindow('MultiTracker')
            
            self.all_selectbox.append(bboxes)
            self.all_selectbox_frame.append(first_frame)
        
        
    def objectNaming(self, dir_name):
        '''
        사용자가 직접 친 bbox의 좌표를 frame에 뿌리면서 객체 이름을 입력 받는다.
        trackerType = 'CSRT' 이므로 CSRT에 해당하는 cv tracker를 bbox, frame 정보와 같이 multiTracker에 add 한다.
        '''

        
        for tracker_info_num in range(len(self.all_selectbox)):
            
            ## Initialize MultiTracker  
            # Create MultiTra cker object
            multiTracker = cv2.MultiTracker_create()
            
            boxes = self.all_selectbox[tracker_info_num]
            box_frame = self.all_selectbox_frame[tracker_info_num]
            
            h_w_info = []
            # Initialize MultiTracker
            # 여기서 ROI영역 보여주고 이름 설정

            roi_set = []
            for box in boxes:
                
                x,y,w,h = box
            
                roi = box_frame[y:y+h, x:x+w]
                roi_set.append([roi, [x,y,w,h]])
                    
                # 트래커에 객체 추가
                multiTracker.add(createTrackerByName(self.trackerType), box_frame, box)

                # h, w 등록
                h_w_info.append(list(roi.shape))
                
            self.multiTracker_list.append(multiTracker)
            
        ##TODO
        # objectnames를 기반으로 object_data.csv 내용이 수정 및 저장되도록
        object_names = []
        for name in self.objectnames:
            if name not in object_names:
                object_names.append(name)

        self.object_df = pd.DataFrame({'DataNames': object_names, 'NumImages': 0.0})
        self.object_df.to_csv(dir_name + '\\object_data.csv', index=False)
        self.object_data = self.object_df['DataNames'].tolist()

    
    def createData(self, capture_list, width, height, dir_name, progressbar):
        
        # 저장되어 있는 데이터 갯수 확인
        train_num, val_num = save_num_check() # 전체 train, valid 데이터 개수 확인
        
        # 데이터 저장 장소 설정
        for multiTracker_num in range(len(self.multiTracker_list)):
            
            if multiTracker_num == 0:   # train video 처리
                save_img_path = dir_name + '\\train'               
                save_name = 'train'            
                num = train_num
            
            elif multiTracker_num == 1: # valid video 처리
                save_img_path =  dir_name + '\\valid'        
                save_name = 'valid'
                num = val_num
            
            multiTracker_ac = self.multiTracker_list[multiTracker_num]
            cap = capture_list[multiTracker_num][0]
            
            label_list = []
            # Process video and track objects
            while cap.isOpened():
                
                success, src = cap.read()
                
                if not success: break

                src_hei, src_wid = src.shape[:2]

                # get updated location of objects in subsequent frames
                # 사용된 tracker이름으로 객체구분
                success, boxes = multiTracker_ac.update(src)
                
                for i, newbox in enumerate(boxes):
                    
                    sleep(0.01)
                    

                    x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])    
                    print(f"TRACKER:{x, y, w, h}")


                    ##TODO: 좌표가 프레임 크기 밖으로 나갔을 때 예외처리
                    if x <= 0 :   # x가 음수라면,
                        w -= abs(x) # 너비에서 절대값 x를 빼준다.
                        x = 0
                        
                    if x >= src_wid: # x가 w 이상 이라면,
                        w = x - w # x는 w보다 크므로 x에서 w를 빼준다.
                        x = src_wid
                        
                    if y <= 0 :
                        h -= abs(y)
                        y = 0
                        
                    if y >= src_hei:
                        h = y - h
                        y = src_hei
                        
                    if x + w > src_wid:
                        w = src_wid - x
                        
                    if y + h > src_hei:
                        h = src_hei - y
                                
                                            
                    ## 확인용 박스 그리기
                    # cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    
                    ## 객체의 인덱스 확인
                    # 기존에 등록된 개체
                    if self.objectnames[i] in self.object_data:
                    
                        label_index = self.object_data.index(f'{self.objectnames[i]}')
                    
                    # 새로운 객체 등록 및 조회
                    elif self.objectnames[i] not in self.object_data:
                        
                        self.object_data = object_insert(self.objectnames[i], 0.0)
                        
                        label_index = self.object_data.index(f'{self.objectnames[i]}')
                    
                    # 객체 info    
                    label_list.append([label_index, 
                                        (float(x) + (float(w) / 2)) / width,   # center x
                                        (float(y) + (float(h) / 2)) / height,  # center y
                                        float(w) / width,                  # width         
                                        float(h) / height,                 # height         
                                        ])
                    
                    sleep(0.01)

                # save img, txt
                save_function(save_img_path, num, src, label_list, save_name)
                        
                label_list.clear()

                # show frame
                #cv2.imshow('MultiTracker', src)
                progressbar.progress_cnt()
                num += 1
                
                # quit on ESC button
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    
                    break
                
            cv2.destroyWindow('MultiTracker')
            
            if multiTracker_num == 0:
            
                print("학습용 데이터 생성이 완료되었습니다.")
            
            elif multiTracker_num ==1:
            
                print("검증용 데이터 생성이 완료되었습니다.")