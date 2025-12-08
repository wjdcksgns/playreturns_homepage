from flask import jsonify
from flask_bcrypt import generate_password_hash, check_password_hash

from flask_jwt_extended import get_jwt
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import create_access_token
from flask_jwt_extended import create_refresh_token
from flask_jwt_extended import set_access_cookies
from flask_jwt_extended import set_refresh_cookies
from flask_jwt_extended import unset_jwt_cookies

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from app import redis
from dao.user import UserDAO

import uuid

user_dao = UserDAO()

# 이메일 중복 확인
def exists(user_email):
  return user_dao.get('email', user_email) is not None

# 회원정보 조회
def get_user(user_id):
  user = user_dao.get('id', user_id)

  if user is None:
    return jsonify({'status': 400, 'message': '회원정보 조회 실패. 회원 없음.'})
  else:
    return jsonify({
      'status': 200,
      'message': '회원정보 조회 성공.',
      'user': {
        'name': user.name,
        'email': user.email,
        'epoch': user.learning_epochs,
        'batch': user.batch_size,
      }
    })

# 회원정보 수정
def patch_user(req, user_id):
  user = user_dao.get('id', user_id)

  if user is None:
    return jsonify({'status': 400, 'message': '회원정보 수정 실패. 회원 없음.'})
  else:
    user_dao.update(user, req)
    
    return jsonify({'status': 200, 'message': '회원정보 수정 성공.'})

# 비밀번호 확인
def check_password(req):
  user = user_dao.get('id', req.get('userId'))

  return user is not None and check_password_hash(user.password, req.get('userPw'))

# 비밀번호 변경
def change_password(req):
  user = user_dao.get('id', req.get('userId'))

  if user is not None and check_password_hash(user.password, req.get('currentPw')):
    user_dao.changePw(user, generate_password_hash(req.get('newPw')).decode('utf-8'))
    
    return jsonify({'status': 200, 'message': '비밀번호 변경 성공.'})

  else:
    status = 400 if user is None else 401
    reason = '회원 없음' if user is None else '비밀번호 불일치'

    return jsonify({'status': status, 'message': f'비밀번호 변경 실패. {reason}.'})

# 회원가입
def join(req):
  try:
    new_user = {
      'uuid': str(uuid.uuid4().hex),
      'name': req.get('userName'),
      'email': req.get('userEmail'),
      'password': generate_password_hash(req.get('userPw')).decode('utf-8')
    }

    user_dao.create(new_user)

    res = jsonify({'status': 200, 'message': '회원가입 성공.'})
  except Exception as e: 
    print(e)
    user_dao.rollback()

    res = jsonify({'status': 400, 'message': '회원가입 실패. 이메일 중복.'})
  finally:
    return res

# 로그인
def login(req):
  user = user_dao.get('email', req.get('userEmail'))

  def is_match():
    return user is not None and check_password_hash(user.password, req.get('userPw'))

  if is_match():
    res = jsonify({
        'status': 200,
        'message': '로그인 성공.',
        'user': {
          'id': user.id,
          'email': user.email,
          'name': user.name
        }
      })

    # token 생성
    access_token = create_access_token(identity=user.email)
    refresh_token = create_refresh_token(identity=user.email)

    # cookie 설정
    set_access_cookies(response=res, encoded_access_token=access_token)
    set_refresh_cookies(response=res, encoded_refresh_token=refresh_token)

    # redis에 설정
    redis.set(refresh_token, user.email, ex=timedelta(days=14))

    return res
  else:
    status = 400 if user is None else 401
    mismatch = '이메일' if user is None else '비밀번호'

    return jsonify({'status': status, 'message': f'로그인 실패. {mismatch} 불일치.'})

# 로그아웃
def logout(req):
  res = jsonify({'status': 200, 'message': '로그아웃 성공.'})

  # redis에 저장되어 있는 refresh token 삭제
  redis.delete(req.cookies.get('refresh_token_cookie'))

  # jwt로 생성된 cookie 전체 삭제
  unset_jwt_cookies(res)

  return res

# jwt token refresh
def refresh(req, user_id):
  token = req.cookies.get('refresh_token_cookie')
  user = user_dao.get('id', user_id)

  # refresh token이 redis에 존재 여부 확인
  has_token = redis.get(token)
  
  # refresh token에 있는 user_email이 유저가 맞는지 확인
  if has_token is None or has_token != user.email:
    return jsonify({'status': 400, 'message': '리프레시 실패.'})
  else:
    # access token 재발급
    res = jsonify({'status': 200, 'message': '리프레시 성공.'})

    current_user = get_jwt_identity()
    
    access_token = create_access_token(identity=current_user)
    set_access_cookies(response=res, encoded_access_token=access_token)

    # refresh token의 expire 시간이 10시간 이하일 경우 refresh token 재발급
    exp_timestamp = get_jwt()['exp']
    now = datetime.now(timezone.utc)
    target_timestamp = datetime.timestamp(now + timedelta(hours=10))

    if target_timestamp > exp_timestamp:
      # 기존 redis에 존재하는 token 삭제
      redis.delete(token)
      refresh_token = create_refresh_token(identity=current_user)
      set_refresh_cookies(response=res, encoded_refresh_token=refresh_token)

      # redis에 토큰 저장
      redis.set(refresh_token, current_user, ex=timedelta(days=14))

    return res