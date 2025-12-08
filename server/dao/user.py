from model.user import User
from app import db

class UserDAO:
  def __init__(self):
    pass

  def get(self, key, value):
    if key == 'id':
      return User.query.get_or_404(value)
    elif key == 'email':
      return User.query.filter_by(email=value).one_or_none()
  
  def create(self, user):
    new_user = User(
      uuid=user['uuid'],
      name=user['name'],
      email=user['email'],
      password=user['password']
    )

    db.session.add(new_user)
    db.session.commit()

  def update(self, user, req):
    if 'name' in req:
      user.name = req['name']
    if 'epoch' in req:
      user.learning_epochs = req['epoch']
    if 'batch' in req:
      user.batch_size = req['batch']

    db.session.commit()
  
  def changePw(self, user, new_pw):
    user.password = new_pw
    
    db.session.commit()
    
  def rollback(self):
    db.session.rollback()