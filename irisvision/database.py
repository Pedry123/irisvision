from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from irisvision.settings import Settings

engine = create_engine(Settings().DATABASE_URL)


def get_session():
    with Session(engine) as session:  # pragma: no cover
        yield session
        # pra sessão não morrer com return para quem chamou,não encerrar antes.
