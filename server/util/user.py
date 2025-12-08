from flask_restx import Namespace, fields

class UserDto:
  User = Namespace(name='User', description='사용자 인증/인가 및 등록 API', path='/user')
  # 사용자 API Request 모델 모음
  LOGIN_INPUTS = User.model('login_request_data', {
    'userEmail': fields.String(description='사용자 이메일', required=True, example='test@coxspace.com'),
    'userPw': fields.String(description='사용자 비밀번호', required=True, example='1234'),
  })
  JOIN_INPUTS = User.inherit('join_request_data', LOGIN_INPUTS, {
    'userName': fields.String(description='사용자 이름', required=True, example='김콕스'),
  })
  # 사용자 API Response 모델 모음
  USER_DATA = User.model('user_information', {
    'id': fields.Integer(description='사용자 유니크 아이디', required=True, example=0),
    'email': fields.String(description='사용자 이메일', required=True, example='test@coxspace.com'),
    'name': fields.String(description='사용자 이름', required=True, example='김콕스'),
  })
  LOGIN_SUCCESS = User.model('login_success_response_data', {
    'status': fields.Integer(description='상태 코드', required=True, example=200),
    'user': fields.Nested(USER_DATA)
  })
  LOGIN_FAILED = User.model('login_error_response_data', {
    'status': fields.Integer(description='상태 코드', required=True, example=400),
    'message': fields.String(description='에러 메세지', required=True, example='로그인 실패. 이메일 혹은 비밀번호 불일치.'),
  })
  JOIN_SUCCESS = User.model('join_success_response_data', {
    'status': fields.Integer(description='상태 코드', required=True, example=200),
    'message': fields.String(description='성공 메세지', required=True, example='회원가입 성공.'),
  })
  JOIN_FAILED = User.model('join_error_response_data', {
    'status': fields.Integer(description='상태 코드', required=True, example=400),
    'message': fields.String(description='에러 메세지', required=True, example='회원가입 실패. 이메일 중복.'),
  })