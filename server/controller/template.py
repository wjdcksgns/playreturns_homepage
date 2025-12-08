from flask import render_template
from util.template import TemplateDto

Template = TemplateDto.Template

@Template.route('/', defaults={'path': ''})
@Template.route('/<path:path>')
def catch_all(path):
  return render_template('index.html')