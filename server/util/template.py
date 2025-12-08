from flask import Blueprint

class TemplateDto:
  Template = Blueprint('template', __name__, url_prefix='/')