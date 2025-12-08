from flask_restx import Namespace, fields

class LabelingServerDto:
  LabelingServer = Namespace (name='Labeling Server', description='데이터 자동 학습 API', path='/labeling-server')