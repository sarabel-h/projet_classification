
import os
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

def convert_pdf_to_images(pdf_path, output_folder=None):
    """
    Convertit chaque page d'un PDF en image.
    Retourne une liste de tuples (index_page, chemin_image_ou_objet_image)
    """
    try:
        images = convert_from_path(pdf_path)
        results = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, img in enumerate(images):
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                img_path = os.path.join(output_folder, f"{base_name}_page_{i+1}.jpg")
                img.save(img_path, 'JPEG')
                results.append(img_path)
            else:
                results.append(img)
        return results
    except Exception as e:
        print(f"Erreur conversion PDF {pdf_path}: {e}")
        return []

def preprocess_image_for_ocr(image_path_or_obj):
    """Nettoyage basique pour améliorer l'OCR"""
    if isinstance(image_path_or_obj, str):
        img = cv2.imread(image_path_or_obj)
    else:
        img = np.array(image_path_or_obj)
        img = img[:, :, ::-1].copy() # RGB to BGR

    # Convertir en gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoising simple
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Binarisation (Otsu)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary)
