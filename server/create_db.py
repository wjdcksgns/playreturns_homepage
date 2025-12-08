import sqlite3

connect = sqlite3.connect('database.db')

print('데이터베이스 생성 성공!')

connect.execute(
  """
  CREATE TABLE user (
    id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    PRIMARY KEY("id" AUTOINCREMENT)
  )
  """
)

print('테이블 생성 성공!')

connect.close()


'''
# DB 초기화 명령어(최초 한번만 실행)
flask db init

# 모델을 새로 생성하거나 변경할 때 사용 (실행하면 작업 파일 생성됨)
flask db migrate

# 모델의 변경 내용을 실제 DB에 적용할 때 사용 (생성된 작업 파일을 실행해 DB 변경)
flask db upgrade
'''