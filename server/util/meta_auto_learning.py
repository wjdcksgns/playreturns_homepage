from flask_restx import Namespace, fields

class MetaAutoLearningDto:
  MetaAutoLearning = Namespace (name='Meta Auto Learning', description='메타 오토 러닝 CRUD API', path='/meta-auto-learning')
  # 메타오토러닝 API Response 모델 모음
  ROW_DATA = MetaAutoLearning.model('my_list_row_response_data', {
    'id': fields.Integer(description='학습 모델 유니크 아이디', required=True, example=0),
    'userId': fields.Integer(description='사용자 유니크 아이디', required=True, example=1),
    'objName': fields.String(description='오브젝트 라벨', required=True, example='cup'),
    'url': fields.Url(description='인식 후 실행할 링크', required=True, example='www.coxspace.com'),
    'tags': fields.List(fields.String(), description='오브젝트 관련 태그 리스트', required=True, default=['cup', '컵']),
    'dataSet': fields.Boolean(description='데이터셋 구축 여부', required=True, example=False),
    'learning': fields.Boolean(description='오브젝트 학습 여부', required=True, example=False),
    'progress': fields.Integer(description='오브젝트 학습률', required=True, example=0),
    'dateTime': fields.DateTime(description='등록 일시', required=True, example='2022-01-01 00:00:00'),
  })