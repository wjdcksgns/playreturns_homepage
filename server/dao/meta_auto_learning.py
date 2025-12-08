from model.meta_auto_learning import VideoObject
from app import db

class VideoObjectDAO:
  def __init__(self):
    pass

  def get(self, key, value):
    if key == 'uuid':
      return VideoObject.query.filter_by(uuid=value).one_or_none()
    elif key == 'exists':
      return VideoObject.query.filter(VideoObject.user_id == value['user_id']).filter(VideoObject.name == value['name']).one_or_none()
  
  def get_all(self, key, value):
    if key == 'user_id':
      return VideoObject.query.filter(VideoObject.user_id == value).order_by(VideoObject.created_at.desc())
  
  def create(self, obj):
    new_obj = VideoObject(
      uuid=obj['uuid'],
      user=obj['user'],
      name=obj['name'],
      url=obj['url'],
      tags=obj['tags'],
      start_x=obj['start_x'],
      start_y=obj['start_y'],
      end_x=obj['end_x'],
      end_y=obj['end_y'],
      color=obj['color'],
      dataset_total_time=obj['dataset_total_time'],
      learning_total_time=obj['learning_total_time'],
      video_url=obj['video_url'],
    )

    db.session.add(new_obj)
    db.session.commit()
  
  def delete(self, obj):
    db.session.delete(obj)
    db.session.commit()

  def rollback(self):
    db.session.rollback()