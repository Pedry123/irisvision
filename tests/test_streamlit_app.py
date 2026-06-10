from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_click_button():
    image_path = Path(__file__).parent / 'images' / 'test1.jpg'
    at = AppTest.from_file('irisvision/ui/app.py').run()
    at.file_uploader[0].upload(
        'test.jpg',
        image_path.read_bytes(),
        'image/jpeg',
    )

    at = at.run()

    assert at.file_uploader[0].value is not None
    assert at.markdown[0].text == '**Prediction:** iris-virginica'
