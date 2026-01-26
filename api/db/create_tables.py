from api.db.db_config import Base, engine

Base.metadata.create_all(bind=engine)

print('✅ Tables created')
