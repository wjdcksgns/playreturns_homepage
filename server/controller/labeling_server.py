from flask import request, send_file, jsonify, current_app
from flask_restx import Resource
from flask_jwt_extended import jwt_required

from sqlalchemy import and_

import base64
import os
import shutil
import cv2
import sys
import torch
import numpy as np
import subprocess
from datetime import datetime
from multiprocessing import Process
from glob import glob
from app import db
from model.user import User
from model.meta_auto_learning import VideoObject

from labeling_server.serverUtil.learnUtil import setUtil
from labeling_server.dataUtil.agumentUtil import agumentutil
from labeling_server.trackerUtil.trackerFuntion import trackerfuntion

from util.labeling_server import LabelingServerDto

LabelingServer = LabelingServerDto.LabelingServer

tracker_function = trackerfuntion()
set_function = setUtil()

def work_dataset(target_obj, progressbar):
  path = target_obj.video_url.split('.mp4')[0]
  train_path = path + '_train_out.mp4'
  valid_path = path + '_valid_out.mp4'

  cap = cv2.VideoCapture(target_obj.video_url)

  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # check w, h
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  
  dir_name = os.path.dirname(target_obj.video_url)

  # read first frame
  fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
  train_out = cv2.VideoWriter(train_path, fourcc, 30.0, (int(width), int(height)))
  valid_out = cv2.VideoWriter(valid_path, fourcc, 30.0, (int(width), int(height)))

  first_frame = None
  for i in range(length):
    success, src = cap.read()
    if i == 0:
      first_frame = src
    if i <= length * 0.8:
      train_out.write(src)
    else:
      valid_out.write(src)

  cap.release()
  train_out.release()
  valid_out.release()

  video_url = [train_path, valid_path]
  capture_list = []

  for i in range(len(video_url)):
    path = video_url[i]
    
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(path)
    
    # read first frame
    success, src = cap.read()
    
    capture_info = [cap, src]
    
    capture_list.append(capture_info)
    
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

  if os.path.exists(dir_name + '\\train') == False:
    os.system('mkdir ' + dir_name + '\\train')

  if os.path.exists(dir_name + '\\valid') == False:
    os.system('mkdir ' + dir_name + '\\valid')

  bboxes = []

  box = tuple([target_obj.start_x, target_obj.start_y, target_obj.end_x - target_obj.start_x, target_obj.end_y - target_obj.start_y])
  bboxes.append(box)
  tracker_function.initialize()

  tracker_function.all_selectbox.append(bboxes)
  tracker_function.all_selectbox.append(bboxes)

  tracker_function.all_selectbox_frame.append(first_frame)
  tracker_function.all_selectbox_frame.append(first_frame)

  tracker_function.objectnames.append(target_obj.name)
  tracker_function.objectnames.append(target_obj.name)

  tracker_function.objectNaming(dir_name)

  # data generation
  tracker_function.createData(capture_list, width, height, dir_name, progressbar)
  
  # data augmentation
  agumentutil(dir_name, progressbar)

## Create training data with video
class ProgressBar:
  def __init__(self):
    self.progressCnt = 0
    self.progressTotal = 0
    self.frameCnt = 0
  def progress_init(self, total):
    self.progressCnt = 0
    self.progressTotal = total
  def progress_status(self):
    status = int((self.progressCnt / self.progressTotal) * 100)
    return status
  def progress_cnt(self):
    self.progressCnt += 1
    status = self.progress_status()

# 영상 오브젝트 데이터셋 구축
@LabelingServer.route('/object/video/dataset', methods = ['PATCH'])
class DatasetBuilding(Resource):
  @jwt_required()
  def patch(self):
    """ 영상 오브젝트 데이터셋 구축 API """
    req = request.get_json()

    def get_target_obj(obj_id):
      return VideoObject.query.filter(VideoObject.uuid == obj_id).first()

    for i in range(len(req['objects'])):
      target_obj = get_target_obj(req['objects'][i]['id'])

      target_obj.dataset_ready = True
      db.session.commit()

    for i in range(len(req['objects'])):
      now = datetime.now()
      target_obj = get_target_obj(req['objects'][i]['id'])
      progressbar = ProgressBar()

      cap = cv2.VideoCapture(target_obj.video_url)
      length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      progressbar.progress_init(length * 2)

      target_obj.dataset_timer = now.timestamp()
      db.session.commit()

      dataset_thread = Process(target=work_dataset, args=(target_obj, progressbar))
      dataset_thread.start()
      dataset_thread.join()

      target_obj.dataset_ready = False
      target_obj.dataset = True
      db.session.commit()
      
    return jsonify({ 'status': 200, 'message': '데이터셋 구축 성공' })

# 영상 오브젝트 학습
@LabelingServer.route('/object/video/learning', methods = ['PATCH'])
class StartLearning(Resource):
  @jwt_required()
  def patch(self):
    req = request.get_json()

    HOST = os.getenv('HOST')
    PORT = os.getenv('SERVER_PORT', 5000)
    MAPS_PATH = 'api/labeling-server/object/video/learning/maps'
    PID_PATH = 'api/labeling-server/object/video/learning/pid'
    maps_url = f'http://{HOST}:{PORT}/{MAPS_PATH}'
    pid_url = f'http://{HOST}:{PORT}/{PID_PATH}'

    obj_uuid_list = []
    train_img_list = []
    valid_img_list = []
    train_path_list = []
    valid_path_list = []
    names_list = []
    dir_name = ''
    nc = len(req['objects'])

    for i in range(len(req['objects'])):
      obj_id = req['objects'][i]['id']
      now = datetime.now()
      
      target_obj = VideoObject.query.filter(VideoObject.uuid == obj_id).first()
      target_obj.learning_ready = True
      target_obj.learning_timer = now.timestamp()

      dir_name = os.path.dirname(target_obj.video_url)
      names_list.append (target_obj.name)
      obj_uuid_list.append(obj_id)
      train_path_list.append(dir_name + '\\train')
      valid_path_list .append(dir_name + '\\valid')
      train_img_list += (glob(dir_name + '\\train' +'\\*.jpg'))
      valid_img_list += (glob(dir_name + '\\valid' +'\\*.jpg'))

    db.session.commit()
    upload_folder = current_app.config['UPLOAD_FOLDER']
    labeling_server_folder = current_app.config['LABELING_SERVER_FOLDER']

    input_size = 640
    batch_size = User.query.get_or_404(req['userId']).batch_size

    weight_pt_filepath = labeling_server_folder + '\\datautil\\model_dataset\\yolov5s.pt'
    train_name = 'result'
    model_config_filepath = labeling_server_folder + '\\datautil\\model_dataset\\yolov5s.yaml'

    train_server_folder = labeling_server_folder + "\\serverUtil\\yolov5\\train.py"

    admin_path = upload_folder +'\\object\\video\\'+ str(req['userId']) +'\\model_dataset'
    epochs = User.query.get_or_404(req['userId']).learning_epochs
    obj_uuid_list, train_path_list, valid_path_list, names_list = ','.join(obj_uuid_list), ','.join(train_path_list), ','.join(valid_path_list), ','.join(names_list)
    proc = Process(target=start_server, args=(train_server_folder, maps_url, pid_url, obj_uuid_list, train_path_list, valid_path_list,names_list,nc,input_size,batch_size,epochs,weight_pt_filepath, train_name, model_config_filepath,admin_path))

    proc.start()
    proc.join()

    return jsonify({ 'status': 200, 'message': '오브젝트 학습 성공' })

def start_server(train_server_folder, maps_url, pid_url, obj_uuid_list, train_path_list, valid_path_list,names_list,nc,input_size,batch_size,epochs,weight_pt_filepath, train_name, model_config_filepath, admin_path):
  output = subprocess.Popen(f'python ' + train_server_folder + f" \
    --maps_url {maps_url} \
    --pid_url {pid_url} \
    --obj_uuid {obj_uuid_list} \
    --train_path {train_path_list} \
    --valid_path {valid_path_list} \
    --names {names_list}\
    --nc {nc} \
    --img {input_size} \
    --batch {batch_size} \
    --epochs {epochs} \
    --weights {weight_pt_filepath} \
    --name {train_name} \
    --cfg {model_config_filepath} \
    --admin_path {admin_path} \
  ", creationflags=subprocess.CREATE_NEW_CONSOLE)
  while output.poll() is None:
    working = True

# 영상 오브젝트 학습 여부 및 학습률 DB 커밋
@LabelingServer.route('/object/video/learning/maps', methods=['POST'])
class GetLearningMaps(Resource):
  def post(self):
    req = request.get_json()
    
    for i, uuid in enumerate(req['uuid']):
      target_obj = VideoObject.query.filter(VideoObject.uuid == uuid).first()
      target_obj.learning_ready = False
      target_obj.learning = True
      target_obj.progress = round(req['maps'][i] * 100, 0)

    db.session.commit()

@LabelingServer.route('/object/video/learning/pid', methods=['POST'])
class GetLearningPID(Resource):
  def post(self):
    req = request.get_json()

    for uuid in req['uuid']:
      target_obj = VideoObject.query.filter(VideoObject.uuid == uuid).first()
      target_obj.learning_pid = req['pid']

    db.session.commit()

# 영상 오브젝트 학습 취소
@LabelingServer.route('/object/video/stop-learning', methods = ['PATCH'])
class StopLearning(Resource):
  @jwt_required()
  def patch(self):
    req = request.get_json()
    user_id = req['userId']

    obj_list = VideoObject.query.filter(and_(VideoObject.user_id==user_id, VideoObject.learning_ready==True)).all()
    obj_cnt = VideoObject.query.filter(and_(VideoObject.user_id==user_id, VideoObject.learning_ready==True)).count()
    upload_folder = current_app.config['UPLOAD_FOLDER']
    model_dataset_path = f'{upload_folder}\\object\\video\\{user_id}\\model_dataset'

    if obj_cnt == len(req['objects']) and os.path.exists(model_dataset_path):
      try:
        process_pid = obj_list[0].learning_pid
        os.system(f'taskkill /f /pid {process_pid}')
        shutil.rmtree(model_dataset_path)

        for target_obj in obj_list:
          target_obj.learning = False
          target_obj.learning_ready = False
          target_obj.learning_timer = None
          target_obj.learning_pid = None
          target_obj.progress = 0
        
        db.session.commit()
        
        res = jsonify({ 'status': 200, 'message': '학습 종료 성공.' })
      except Exception as e: 
        print(e)
        db.session.rollback()

        res = jsonify({ 'status': 401, 'message': '학습 종료 실패. 프로세스 또는 디렉토리 없음.' })
      finally:
        return res
    else:
      return jsonify({ 'status': 400, 'message': '학습 종료 실패. 오브젝트 불일치.' })
        
##파일 다운로드 처리
@LabelingServer.route('/fileDown', methods = ['POST'])
class DownFile(Resource):
  def post(self):
    if request.method == 'POST':
      
      sw=0
      files = os.listdir("./dataUtil/model_weight_result")
      for x in files:
        
        if(x==request.form['file']):
          sw=1

      path = "./dataUtil/model_weight_result/" 
      return send_file(path + request.form['file'],
        attachment_filename = request.form['file'],
        as_attachment=True)

## detect.py 처리
@LabelingServer.route('/object/detection', methods = ['POST'])
class StartDetect(Resource):
  @jwt_required()
  def post(self):
    req = request.get_json()

    upload_folder = current_app.config['UPLOAD_FOLDER']
    labeling_server_folder = current_app.config['LABELING_SERVER_FOLDER']

    yolov5_path = labeling_server_folder + '\\serverUtil\\yolov5'
    model_pt_path = upload_folder +'\\object\\video\\'+ str(req['userId']) +'\\model_dataset\\result\\weights\\best.pt'

    if os.path.exists(model_pt_path):
      custom_model = torch.hub.load(yolov5_path, 'custom', path=model_pt_path, source='local')

      decoded_data = base64.b64decode(req['imageBlob'])
      np_data = np.fromstring(decoded_data, np.uint8)
      img = cv2.imdecode(np_data,cv2.IMREAD_COLOR)

      results = custom_model(img)

      dic_list = []

      none = False

      for pred in results.pred[0]:
        name = results.names[int(pred[-1])] # 객체 이름 추출
        target_obj = VideoObject.query.filter(VideoObject.user_id == req['userId']).filter(VideoObject.name == name).first()
        url = target_obj.url
        color = target_obj.color

        # 좌표 추출
        start_x = pred[0]
        start_y = pred[1]
        end_x = pred[2]
        end_y = pred[3]

        if pred[4] < 0.7:
          none = True
          break
        
        if any(v['parentname'] == name for v in dic_list) is False :
          dic_list.append({
              "parentname": f"{name}",
              "rec_url": f"{url}",
              "rec_color": f"{color}",
              "rec_start_x" : f'{start_x}',
              "rec_start_y" : f'{start_y}',
              "rec_end_x" : f'{end_x}',
              "rec_end_y" : f'{end_y}'
          })

      if none == True:
        dic_list = []

      return jsonify({ 'status': 200, 'message': 'Detection 성공.', 'data': dic_list })
    else:
      return jsonify({ 'status': 400, 'message': 'Detection 실패. PT 파일 없음.' })