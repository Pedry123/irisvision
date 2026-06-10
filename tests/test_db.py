from dataclasses import asdict

from sqlalchemy import select

from irisvision.models import Classification


def test_create_classification(session, mock_db_time):
    with mock_db_time(model=Classification) as time:
        classif = Classification(prediction='iris-setosa', confidence=0.784342)

        session.add(classif)
        session.commit()

        result = session.scalar(
            select(Classification).where(
                Classification.prediction == 'iris-setosa'
            )
        )

        assert asdict(result) == {
            'prediction': 'iris-setosa',
            'confidence': 0.784342,
            'id': 1,
            'created_at': time,
            'updated_at': time,
        }
