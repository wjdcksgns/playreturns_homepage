from flask import request
from flask_restx import Resource
from flask_jwt_extended import jwt_required

from util.user import UserDto
from service.user import exists, get_user, patch_user, check_password, change_password, join, login, logout, refresh

User = UserDto.User

# 이메일 중복 확인
@User.route('/exists/<user_email>', methods=['GET'])
class Exists(Resource):
  def get(self, user_email):
    """ 사용자 이메일 중복 확인 API """
    return exists(user_email)

# 회원정보 조회/수정
@User.route('/<user_id>', methods=['GET', 'PATCH'])
class UserInfo(Resource):
  @jwt_required()
  def get(self, user_id):
    """ 사용자 정보 조회 API """
    return get_user(user_id)
  @jwt_required()
  def patch(self, user_id):
    """ 사용자 정보 수정 API """
    return patch_user(request.get_json(), user_id)

# 비밀번호 확인
@User.route('/check/password', methods=['POST'])
class CheckPassword(Resource):
  @jwt_required()
  def post(self):
    """ 비밀번호 확인 API """
    return check_password(request.get_json())

# 비밀번호 변경
@User.route('/password', methods=['PUT'])
class UserPassword(Resource):
  @jwt_required()
  def put(self):
    """ 비밀번호 변경 API """
    return change_password(request.get_json())

# 회원가입
@User.route('/join', methods=['POST'])
@User.expect(UserDto.JOIN_INPUTS)
class Join(Resource):
  @User.response(200, '회원가입 성공', UserDto.JOIN_SUCCESS)
  @User.response(400, '회원가입 실패 (이메일 중복)', UserDto.JOIN_FAILED)
  def post(self):
    """ 사용자 회원가입 API """
    return join(request.get_json())

# 로그인
@User.route('/login', methods=['POST'])
@User.expect(UserDto.LOGIN_INPUTS)
class Login(Resource):
  @User.response(200, '로그인 성공', UserDto.LOGIN_SUCCESS)
  @User.response(400, '로그인 실패 (이메일 혹은 비밀번호 불일치)', UserDto.LOGIN_FAILED)
  def post(self):
    """ 사용자 로그인 API """
    return login(request.get_json())

# 로그아웃
@User.route('/logout', methods=['DELETE'])
class Logout(Resource):
  @jwt_required()
  def delete(self):
    """ 사용자 로그아웃 API """
    return logout(request)

# 리프레시
@User.route('/refresh/<user_Id>', methods=['GET'])
class Refresh(Resource):
  @jwt_required(refresh=True)
  def get(self, user_Id):
    """ 리프레시 토큰 API """
    return refresh(request, user_Id)