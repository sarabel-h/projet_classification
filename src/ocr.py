import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def extract_text(path):
    """
    Extract text from PDF or image using OCR (French + Arabic)
    """
    text = ""

    if path.lower().endswith(".pdf"):
        images = convert_from_path(path)
        for img in images:
            page_text = pytesseract.image_to_string(
                img,
                lang="fra+ara",
                config="--psm 6"
            )
            text += "\n" + page_text

    else:
        img = Image.open(path)
        text = pytesseract.image_to_string(
            img,
            lang="fra+ara",
            config="--psm 6"
        )

    return text.lower()
