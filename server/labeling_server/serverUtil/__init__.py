import os
import cv2
import sys
from .learnUtil import setUtil
from .yolov5.train import Train
from .config import Config as cfg
from flask import Flask, request, send_file
from labeling_server.dataUtil.objectDataCheck import object_check
from labeling_server.dataUtil.agumentUtil import agumentutil
from labeling_server.trackerUtil.trackerFuntion import trackerfuntion

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__)
app.debug = False
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #파일 업로드 용량 제한 단위:바이트


## Temporary video storage location
# Train setting
trainPath = "./dataUtil/tracker_data/train_tracker/test1.mp4"
traindataPath = "./dataUtil/model_dataset/model_learn_images/train"

# val setting
valPath = './dataUtil/tracker_data/validation_tracker/test2.mp4'
valdataPath = "./dataUtil/model_dataset/model_learn_images/valid"

# Data path to import, data storage path
video_list = [trainPath, valPath]
save_path_list = [traindataPath, valdataPath]

tracker_funtion = trackerfuntion()
set_funtion = setUtil()

##TODO: 객체 좌표, 동영상 첫 이미지 받는 route

## Create training data with video
@app.route('/select_object_page/', methods = ['POST', 'GET'])
def select_object():
    
    if request.method == 'POST':
        
        capture_list = []

        # Import registered videos in order
        for i in range(len(video_list)):
            
            path = video_list[i]
            
            # Create a video capture object to read videos
            cap = cv2.VideoCapture(path)
            
            # check w, h
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # read first frame
            success, src = cap.read()
            
            capture_info = [cap, src]
            
            capture_list.append(capture_info)
            
            # quit if unable to read the video file
            if not success:
                print('Failed to read video')
                sys.exit(1)
                
        """-------------------------------------------------------------------"""

        # Processing each object's bouncing box
        tracker_funtion.selectObject(capture_list)

        """-------------------------------------------------------------------"""

        # Determining object names for each bounding box
        # create multitracker
        tracker_funtion.objectNaming()

        """-------------------------------------------------------------------"""
        
        # data generation
        tracker_funtion.createData(capture_list, width, height)
        print("데이터 생성이 완료되었습니다.")
        
        """-------------------------------------------------------------------"""

        # data augmentation
        agumentutil()
        print("데이터 증식이 완료되었습니다.")
        

##Start Learning
@app.route('/learning_page', methods = ['POST', 'GET'])
def startLearning():
    
    if request.method == 'POST':
        
        object_list = request.form.getlist('object_name', None)
        batch_size = request.args.get('batch_size')
        epochs = request.args.get('epochs')
        learn_type = request.args.get('type')
        

        if learn_type == 0:
            # only input now object - img path set
            set_funtion.img_set_util()

        ##TODO: 이미지 전체 저장한 디렉토리와 현재 생성된 이미지 디렉토리 나누기
        elif learn_type == 1:
            # all object - img path set
            set_funtion.img_set_util()

        if learn_type == 0:
            # only input now object-yaml file set
            set_funtion.yaml_upload(object_list)

        elif learn_type == 1:
            # all object - yaml file set
            set_funtion.yaml_upload(object_check())


        print('Start training the Model')
        
        os.chdir('yolov5')
        os.system(f"python train.py \
            --img {cfg.input_size} \
            --batch {batch_size} \
            --epochs {epochs} \
            --data {cfg.data_yaml} \
            --weights {cfg.weight_pt} \
            --name {cfg.train_name} \
            --cfg {cfg.model_config}")
        
        ##TODO: 계속 return if 학습 데이터 갯수 100%이면 .pt파일 반환

        ##TODO: .pt파일 반환 후 만들어진 파일 삭제


## Learning Total Percentage
@app.route('/present_progress', methods = ['POST', 'GET'])
def get_persent():
    
    if request.method == 'POST':
        
        return Train.get_train_info()
        
        
##파일 다운로드 처리
@app.route('/fileDown', methods = ['GET', 'POST'])
def down_file():
	if request.method == 'POST':
     
		sw=0
		files = os.listdir("./dataUtil/model_weight_result")
		for x in files:
      
			if(x==request.form['file']):
				sw=1

		path = "./dataUtil/model_weight_result/" 
		return send_file(path + request.form['file'],           # 다운 받을 파일 (경로포함)
				attachment_filename = request.form['file'],     # 다운 받아지는 파일 이름
				as_attachment=True)
        

if __name__ == '__main__':
    app.run(debug=False, threaded = True)