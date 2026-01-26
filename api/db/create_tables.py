from api.db.db_config import engine, Base
from api.db import models


Base.metadata.create_all(bind=engine)

print('✅ Tables created')