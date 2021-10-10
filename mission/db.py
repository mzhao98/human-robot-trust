from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# user = "root"
# password = "Turbotax98!"
# host = "localhost:3306"
# db_name = "sar_game"
user = "admin"
password = "michelle"
host = "testdb.cqnxcsogehau.us-east-1.rds.amazonaws.com"
db_name = "sargame"

DATABASE_URL = 'mysql+mysqlconnector://%s:%s@%s/%s?charset=utf8' % (
    user,
    password,
    host,
    db_name,
)

ENGINE = create_engine(
    DATABASE_URL,
    encoding="utf-8"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)

Base = declarative_base()
