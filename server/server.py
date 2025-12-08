from waitress import serve
from app import create_app

import os

ENV_MODE = os.getenv('ENV_MODE', 'dev')

app = create_app(ENV_MODE)
print(app.config['MESSAGE'])

if __name__ == '__main__':
  HOST = os.getenv('REDIS_HOST', '0.0.0.0')
  PORT = os.getenv('SERVER_PORT', 5000)

  if ENV_MODE == "prod":
    serve(app, host=HOST, port=PORT)
  else:
    app.run(host=HOST, port=PORT)