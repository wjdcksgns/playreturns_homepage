from datetime import timedelta
from dotenv import load_dotenv

import os

# 환경변수
load_dotenv()

HOST = os.getenv('HOST')
REDIS_HOST = os.getenv('REDIS_HOST')

class Config(object):
  SITE = f'http://{HOST}:5000'
  REDIS_URL = f'redis://{HOST}:6379'
  DEBUG = False
  TESTING = False
  PROPAGATE_EXCEPTIONS = True
  
  SECRET_KEY = os.getenv('SECRET_KEY')
  BCRYPT_LEVEL = 10

  BASE_DIR = os.path.dirname(__file__)

  SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'database.db'))
  SQLALCHEMY_TRACK_MODIFICATIONS = False

  JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
  JWT_DECODE_ALGORITHMS = ['HS256']
  JWT_TOKEN_LOCATION = ['cookies']
  JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=10)
  JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=14)
  JWT_COOKIE_SECURE = False
  JWT_COOKIE_CSRF_PROTECT = True
  JWT_CSRF_METHODS = ['POST', 'PUT', 'PATCH', 'DELETE', 'GET']
  JWT_CSRF_CHECK_FORM = True
  JWT_CSRF_IN_COOKIES = True

  UPLOAD_FOLDER = '.\\upload'
  LABELING_SERVER_FOLDER = '.\\labeling_server'

class ProductionConfig(Config):
  MESSAGE = 'Product'
  JWT_COOKIE_SECURE = True

class DevelopmentConfig(Config):
  MESSAGE = 'Development'
  DEBUG = True

class TestingConfig(Config):
  MESSAGE = 'Testing'
  TESTING = True

config_by_name = dict(
  dev=DevelopmentConfig,
  test=TestingConfig,
  prod=ProductionConfig
)