from app import db
from sqlalchemy.sql import func
from sqlalchemy import CheckConstraint

class User(db.Model):
  id = db.Column(db.Integer, nullable=False, primary_key=True, unique=True, autoincrement=True)
  uuid = db.Column(db.String(200), nullable=False, unique=True)
  name = db.Column(db.String(200), nullable=False)
  email = db.Column(db.String(200), nullable=False, unique=True)
  password = db.Column(db.String(200), nullable=False)
  learning_epochs = db.Column(db.Integer, default=20, nullable=False)
  batch_size = db.Column(db.Integer, default=4, nullable=False)
  created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
  updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())
  
  __table_args__ = (
    CheckConstraint('batch_size >= 2', name='batch_size_min'),
    CheckConstraint('batch_size <= 32', name='batch_size_max'),
    CheckConstraint('learning_epochs >= 20', name='learning_epochs_min'),
    CheckConstraint('learning_epochs <= 500', name='learning_epochs_max'),
  )