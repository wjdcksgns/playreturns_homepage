from app import db
from sqlalchemy.sql import func

class VideoObject(db.Model):
  id = db.Column(db.Integer, nullable=False, primary_key=True, unique=True, autoincrement=True)
  uuid = db.Column(db.String(200), nullable=False, unique=True)

  # user 참조
  user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'))
  user = db.relationship('User', backref=db.backref('video_object', cascade='all, delete-orphan'))
  
  # object
  name = db.Column(db.String(200), nullable=False)
  url = db.Column(db.String(200), nullable=False)
  tags = db.Column(db.String(200), nullable=False)

  # rectangle
  start_x = db.Column(db.Integer, nullable=False)
  start_y = db.Column(db.Integer, nullable=False)
  end_x = db.Column(db.Integer, nullable=False)
  end_y = db.Column(db.Integer, nullable=False)
  color = db.Column(db.String(30), nullable=False)

  # 학습
  dataset = db.Column(db.Boolean, default=False, nullable=False)
  dataset_ready = db.Column(db.Boolean, default=False, nullable=False)
  dataset_timer = db.Column(db.String(200))
  dataset_total_time = db.Column(db.String(200), nullable=False)
  learning = db.Column(db.Boolean, default=False, nullable=False)
  learning_ready = db.Column(db.Boolean, default=False, nullable=False)
  learning_timer = db.Column(db.String(200))
  learning_total_time = db.Column(db.String(200), nullable=False)
  learning_pid = db.Column(db.String(50), nullable=True)
  progress = db.Column(db.Integer, default=0, nullable=False)
  
  # 파일
  video_url = db.Column(db.String(255), nullable=False)

  created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
  updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())