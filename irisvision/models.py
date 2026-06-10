from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, registry

table_registry = registry()


@table_registry.mapped_as_dataclass
class Classification:
    __tablename__ = 'classifications'

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    prediction: Mapped[str] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now(), onupdate=func.now()
    )
