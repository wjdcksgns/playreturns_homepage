from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

import redis
import os.path

# 환경변수
from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv('HOST')

api = Api(version='0.1', title='Meta Space API Server', description='메타 스페이스 API 문서입니다.', doc='/documentation', prefix='/api')
cors = CORS()
db = SQLAlchemy()
migrate = Migrate()
flask_bcrypt = Bcrypt()
jwt = JWTManager()

redis = redis.StrictRedis(host=HOST, port=6379, db=0, decode_responses=True)

# Router 등록
def register_router(flask_app: Flask):
  # Pages
  from controller.template import Template
  flask_app.register_blueprint(Template)

  # Favicon
  from controller.favicon import Favicon
  flask_app.register_blueprint(Favicon)


# API 등록
def register_api(flask_api):
  # 사용자 인증/인가 및 등록 API
  from controller.user import User
  flask_api.add_namespace(User)

  # 메타 오토 러닝 CRUD API
  from controller.meta_auto_learning import MetaAutoLearning
  flask_api.add_namespace(MetaAutoLearning)

  # 데이터 자동 학습 API
  from controller.labeling_server import LabelingServer
  flask_api.add_namespace(LabelingServer)


# 앱 설정
def create_app(mode='dev'):
  app = Flask(__name__, template_folder='build', static_folder='build/static')

  api.init_app(app)
  if mode == 'dev':
    cors.init_app(app, supports_credentials=True, resources={r'*': {"origins": "*"}})
  else:
    cors.init_app(app, supports_credentials=True, resources={r"/api/*": {'origins': f'http://{HOST}:3000'}})
  flask_bcrypt.init_app(app)
  jwt.init_app(app)

  # Config
  from config import config_by_name
  app.config.from_object(config_by_name[mode])

  # ORM
  db.init_app(app)
  migrate.init_app(app, db)

  from model import user, meta_auto_learning

  register_router(app)
  register_api(api)

  return app