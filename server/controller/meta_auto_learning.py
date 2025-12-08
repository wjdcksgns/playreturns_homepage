from flask import request
from flask_restx import Resource
from flask_jwt_extended import jwt_required

from util.meta_auto_learning import MetaAutoLearningDto
from service.meta_auto_learning import my_list, exists, add_video, del_video

MetaAutoLearning = MetaAutoLearningDto.MetaAutoLearning

# 마이리스트 조회
@MetaAutoLearning.route('/my-list/<int:user_id>', methods=['GET'])
@MetaAutoLearning.doc(params={'user_id': '사용자 유니크 아이디'})
class MyList(Resource):
  @MetaAutoLearning.response(200, '조회 성공', [MetaAutoLearningDto.ROW_DATA])
  @jwt_required()
  def get(self, user_id):
    """ 마이리스트 조회 API """
    return my_list(user_id)

# 오브젝트 이름 중복 확인
@MetaAutoLearning.route('/object/exists/<int:user_id>/<object_name>', methods=['GET'])
class ExistsObject(Resource):
  @jwt_required()
  def get(self, user_id, object_name):
    """ 오브젝트 이름 중복 확인 API """
    return exists(user_id, object_name)

# 영상 오브젝트 추가
@MetaAutoLearning.route('/object/video', methods=['POST'])
class CreateObject(Resource):
  @jwt_required()
  def post(self):
    """ 영상 오브젝트 추가 API """
    return add_video(request)

# 영상 오브젝트 삭제
@MetaAutoLearning.route('/object/video/<string:object_uuid>', methods=['DELETE'])
class DeleteObject(Resource):
  @jwt_required()
  def delete(self, object_uuid):
    """ 영상 오브젝트 삭제 API """
    return del_video(object_uuid)