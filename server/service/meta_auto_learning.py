from flask import jsonify, current_app
from werkzeug.utils import secure_filename

from datetime import datetime

from dao.user import UserDAO
from dao.meta_auto_learning import VideoObjectDAO

import os
import shutil
import pandas
import json
import cv2
import uuid

user_dao = UserDAO()
video_dao = VideoObjectDAO()

# 마이리스트 조회
def my_list(user_id):
  user = user_dao.get('id', user_id)
  
  if user is None:
    return jsonify({'status': 400, 'message': '마이리스트 조회 실패. 사용자 없음.'})
  else:
    sorted_list = video_dao.get_all('user_id', user_id)
    data_frame = pandas.read_sql(sorted_list.statement, sorted_list.session.connection())
    my_list = json.loads(data_frame.to_json(orient='records'))

    def get_new_list(v):
      remain = 0
      percent = 0
      if v['dataset']:
        dataset = 100
      else:
        if v['dataset_ready']:
          if v['dataset_timer'] is None:
            dataset = -1
          else:
            now = datetime.now()
            total = float(v['dataset_total_time'])
            timer = float(v['dataset_timer']) + total
            remain = timer - now.timestamp()

            if remain <= 0:
              remain = 1
            percent = 100 - int((remain / total) * 100)
            dataset = 99 if percent > 99 else percent
        else:
          dataset = 0

      learning_total = float(v['learning_total_time']) * float(user.learning_epochs)
      return {
        'uuid': v['uuid'],
        'name': v['name'],
        'datetime': v['created_at'],
        'tags': v['tags'],
        'url': v['url'],
        'dataset': dataset,
        'dataset_total': v['dataset_total_time'],
        'learning_total': learning_total,
        'learning_ready': v['learning_ready'],
        'learning_timer': v['learning_timer'],
        'learning': v['learning'],
        'progress': v['progress']
      }
    
    new_list = list(map(get_new_list , my_list))

    return jsonify({
      'status': 200,
      'message': '마이리스트 조회 성공.',
      'data': None if not my_list else new_list
    })

# 오브젝트 이름 중복 확인
def exists(user_id, name):
  return video_dao.get('exists', {'user_id': user_id, 'name': name}) is not None

# 영상 오브젝트 추가
def add_video(request):
  req = request.form

  # 업로드 폴더 지정
  upload_folder = current_app.config['UPLOAD_FOLDER']
  user = user_dao.get('id', req['userId'])
  uuid_hex = str(uuid.uuid4().hex)
  save_location = f'{upload_folder}\\object\\video\\{user.id}\\{uuid_hex}'

  if os.path.exists(save_location) is False:
    os.system('mkdir ' + save_location)

  # 업로드 파일 지정
  video = request.files['file']
  video_name = secure_filename(video.filename)
  datetime_format = datetime.now().strftime('%Y%m%d%H%M%S%f')
  save_video_name = f'{datetime_format}_{video_name}'
  video_url = f'{save_location}\\{save_video_name}'

  if os.path.exists(video_url) is False:
    # 비디오 파일 저장
    video.save(os.path.join(save_location, save_video_name))

    # 비디오 파일 사이즈
    cap = cv2.VideoCapture(video_url)
    cap_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()

    dataset_building_sec = 40
    frame_len = (cap_length * 2)
    dataset_one_frame_sec = dataset_building_sec / 60
    dataset_total_sec = int(frame_len * dataset_one_frame_sec)
    learning_total_sec = dataset_total_sec * dataset_one_frame_sec

    if ret:
      # 좌표값 비율 계산
      canvas_size = json.loads(req['canvasSize'])
      rectangle = json.loads(req['rectangle'])
      ratio_x = cap_width / canvas_size['width']
      ratio_y = cap_height / canvas_size['height']
      start_x = int(rectangle['startX'] * ratio_x)
      start_y = int(rectangle['startY'] * ratio_y)
      end_x = int(rectangle['endX'] * ratio_x)
      end_y = int(rectangle['endY'] * ratio_y)

      # DB에 데이터 저장
      try:
        new_video_obj = {
          'uuid': uuid_hex,
          'user': user,
          'name': req['objName'],
          'url': req['url'],
          'tags': req['tags'],
          'start_x': start_x,
          'start_y': start_y,
          'end_x': end_x,
          'end_y': end_y,
          'color': req['color'],
          'dataset_total_time': str(dataset_total_sec),
          'learning_total_time': str(learning_total_sec),
          'video_url': video_url,
        }

        video_dao.create(new_video_obj)

        res = jsonify({'status': 200, 'message': '오브젝트 추가 성공.'})
      except Exception as e: 
        print(e)
        video_dao.rollback()

        res = jsonify({'status': 400, 'message': '오브젝트 추가 실패.'})
      finally:
        return res

# 영상 오브젝트 삭제
def del_video(object_uuid):
  target_obj = video_dao.get('uuid', object_uuid)

  if target_obj is None:
    return jsonify({'status': 400, 'message': '오브젝트 삭제 실패. 오브젝트 없음.'})
  else:
    dir_path = os.path.dirname(target_obj.video_url)

    if os.path.exists(dir_path):
      shutil.rmtree(dir_path)
      video_dao.delete(target_obj)

      return jsonify({'status': 200, 'message': '오브젝트 삭제 성공.'})
    else:
      return jsonify({'status': 401, 'message': '오브젝트 삭제 실패. 디렉토리 없음.'})