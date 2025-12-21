"""
Module de prétraitement des documents (PDF -> images -> texte)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("pdf2image not available. PDF support disabled.")

try:
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    logging.warning("pytesseract not available. OCR support disabled.")

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Prétraite les documents (PDF, images) pour la classification"""
    
    def __init__(self, dpi: int = 300):
        """
        Initialise le préprocesseur
        
        Args:
            dpi: Résolution pour la conversion PDF (par défaut 300)
        """
        self.dpi = dpi
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convertit un fichier PDF en images
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Liste des images (numpy arrays)
        """
        if not PDF_SUPPORT:
            logger.error("pdf2image n'est pas installé")
            return []
        
        try:
            logger.info(f"Conversion du PDF: {pdf_path}")
            images = convert_from_path(pdf_path, dpi=self.dpi)
            
            # Convertir en numpy arrays
            images_np = [np.array(img) for img in images]
            logger.info(f"[OK] {len(images_np)} pages converties")
            
            return images_np
        
        except Exception as e:
            logger.error(f"Erreur lors de la conversion du PDF: {e}")
            return []
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Charge une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Impossible de charger: {image_path}")
                return None
            
            logger.info(f"[OK] Image chargée: {image_path}")
            return image
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Redimensionne une image
        
        Args:
            image: Image numpy
            target_size: Taille cible (hauteur, largeur)
            
        Returns:
            Image redimensionnée
        """
        return cv2.resize(image, (target_size[1], target_size[0]))
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise une image [0, 255] -> [0, 1]
        
        Args:
            image: Image numpy
            
        Returns:
            Image normalisée
        """
        return image.astype(np.float32) / 255.0
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Réduit le bruit dans une image
        
        Args:
            image: Image numpy
            
        Returns:
            Image débruitée
        """
        if len(image.shape) == 3:
            image_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            image_denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return image_denoised
    
    def adjust_contrast(self, image: np.ndarray, alpha: float = 1.2, beta: float = 30) -> np.ndarray:
        """
        Ajuste le contraste d'une image
        
        Args:
            image: Image numpy
            alpha: Facteur de contraste (>1 augmente)
            beta: Facteur de luminosité
            
        Returns:
            Image avec contraste ajusté
        """
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def straighten_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redresse une image légèrement penchée
        
        Args:
            image: Image numpy
            
        Returns:
            Image redressée
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter les contours
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return image
        
        # Calculer l'angle moyen
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        if not angles:
            return image
        
        median_angle = np.median(angles)
        
        if abs(median_angle) < 1:  # Correction mineure
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            image_straight = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            return image_straight
        
        return image
    
    def extract_text_with_ocr(self, image: np.ndarray, lang: str = 'fra') -> str:
        """
        Extrait le texte d'une image avec OCR
        
        Args:
            image: Image numpy
            lang: Code langue (fra pour français)
            
        Returns:
            Texte extrait
        """
        if not OCR_SUPPORT:
            logger.error("pytesseract n'est pas installé")
            return ""
        
        try:
            # Prétraiter pour améliorer l'OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # OCR
            text = pytesseract.image_to_string(binary, lang=lang)
            
            logger.info(f"✓ Texte extrait ({len(text)} caractères)")
            return text
        
        except Exception as e:
            logger.error(f"Erreur lors de l'OCR: {e}")
            return ""
    
    def preprocess_for_cv(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Prétraite une image pour le computer vision
        
        Args:
            image: Image numpy
            target_size: Taille cible
            
        Returns:
            Image prétraitée
        """
        # Débruiter
        image = self.denoise_image(image)
        
        # Ajuster le contraste
        image = self.adjust_contrast(image)
        
        # Redresser
        image = self.straighten_image(image)
        
        # Redimensionner
        image = self.resize_image(image, target_size)
        
        # Normaliser
        image = self.normalize_image(image)
        
        return image
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite une image pour l'OCR
        
        Args:
            image: Image numpy
            
        Returns:
            Image prétraitée
        """
        # Débruiter
        image = self.denoise_image(image)
        
        # Ajuster le contraste
        image = self.adjust_contrast(image, alpha=1.5, beta=50)
        
        # Redresser
        image = self.straighten_image(image)
        
        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = DocumentPreprocessor()
    logger.info("DocumentPreprocessor initialized")
