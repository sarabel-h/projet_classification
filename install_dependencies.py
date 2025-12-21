"""
Script d'installation des dépendances système pour Windows
Installe Poppler et les fichiers de langue Tesseract
"""

import os
import sys
import urllib.request
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, dest_path):
    """Télécharge un fichier depuis une URL"""
    logger.info(f"Téléchargement depuis {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"[OK] Téléchargé: {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        return False


def install_poppler():
    """
    Guide d'installation de Poppler pour Windows
    """
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION DE POPPLER (pour support PDF)")
    logger.info("="*60)
    
    print("""
Poppler est nécessaire pour la conversion PDF vers images.

INSTALLATION MANUELLE (Recommandée):
===================================

1. Téléchargez Poppler pour Windows:
   https://github.com/oschwartz10612/poppler-windows/releases/
   
2. Choisissez la dernière version (ex: poppler-24.08.0.zip)

3. Extrayez le ZIP dans: C:\\Program Files\\poppler\\

4. Ajoutez au PATH Windows:
   C:\\Program Files\\poppler\\Library\\bin

5. Redémarrez votre terminal

VÉRIFICATION:
=============
Après installation, testez avec:
    python -c "from pdf2image import convert_from_path; print('PDF support OK')"
""")
    
    response = input("\nAppuyez sur Entrée pour continuer...")
    

def install_tesseract_french():
    """
    Télécharge les fichiers de langue française pour Tesseract
    """
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION DES LANGUES TESSERACT (Français)")
    logger.info("="*60)
    
    tessdata_dir = Path("C:\\Program Files\\Tesseract-OCR\\tessdata")
    
    if not tessdata_dir.exists():
        logger.error(f"Tesseract n'est pas installé dans: {tessdata_dir}")
        logger.info("\nInstallez d'abord Tesseract depuis:")
        logger.info("https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    # URLs des fichiers de langue
    base_url = "https://github.com/tesseract-ocr/tessdata/raw/main/"
    french_files = ["fra.traineddata", "osd.traineddata"]
    
    for lang_file in french_files:
        dest_file = tessdata_dir / lang_file
        
        if dest_file.exists():
            logger.info(f"[OK] {lang_file} déjà installé")
            continue
        
        url = base_url + lang_file
        logger.info(f"Téléchargement de {lang_file}...")
        
        try:
            urllib.request.urlretrieve(url, dest_file)
            logger.info(f"[OK] {lang_file} installé")
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de {lang_file}: {e}")
            logger.info(f"\nTéléchargez manuellement depuis: {url}")
            logger.info(f"Et placez-le dans: {tessdata_dir}")
            return False
    
    logger.info("\n[OK] Fichiers de langue français installés!")
    return True


def verify_installations():
    """Vérifie que toutes les dépendances sont installées"""
    logger.info("\n" + "="*60)
    logger.info("VÉRIFICATION DES INSTALLATIONS")
    logger.info("="*60)
    
    # Vérifier pytesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"[OK] Tesseract version: {version}")
    except Exception as e:
        logger.error(f"[ERREUR] Tesseract: {e}")
    
    # Vérifier les langues Tesseract
    try:
        import pytesseract
        result = pytesseract.get_languages()
        if 'fra' in result:
            logger.info("[OK] Langue française disponible")
        else:
            logger.warning("[ATTENTION] Langue française non disponible")
            logger.info(f"Langues disponibles: {result}")
    except Exception as e:
        logger.error(f"[ERREUR] Langues Tesseract: {e}")
    
    # Vérifier pdf2image/poppler
    try:
        from pdf2image import convert_from_path
        logger.info("[OK] pdf2image installé")
        # Essayer de convertir un PDF fictif pour tester Poppler
        logger.info("[INFO] Poppler doit être installé pour la conversion PDF")
    except Exception as e:
        logger.error(f"[ERREUR] pdf2image: {e}")
    
    # Vérifier OpenCV
    try:
        import cv2
        logger.info(f"[OK] OpenCV version: {cv2.__version__}")
    except Exception as e:
        logger.error(f"[ERREUR] OpenCV: {e}")
    
    # Vérifier PyTorch
    try:
        import torch
        logger.info(f"[OK] PyTorch version: {torch.__version__}")
    except Exception as e:
        logger.error(f"[ERREUR] PyTorch: {e}")
    
    # Vérifier Transformers
    try:
        import transformers
        logger.info(f"[OK] Transformers version: {transformers.__version__}")
    except Exception as e:
        logger.error(f"[ERREUR] Transformers: {e}")


def main():
    """Installation principale"""
    logger.info("="*60)
    logger.info("INSTALLATION DES DÉPENDANCES SYSTÈME")
    logger.info("="*60)
    
    print("""
Ce script va vous guider pour installer:
1. Poppler (pour support PDF)
2. Fichiers de langue Tesseract (français)

PRÉREQUIS:
- Tesseract OCR doit être installé
- Connexion internet active
""")
    
    choice = input("\nContinuer? (o/n): ").lower()
    
    if choice != 'o':
        logger.info("Installation annulée")
        return
    
    # 1. Guide Poppler
    install_poppler()
    
    # 2. Installer langue française Tesseract
    install_tesseract_french()
    
    # 3. Vérifications
    verify_installations()
    
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION TERMINÉE")
    logger.info("="*60)
    logger.info("\nRedémarrez votre terminal et testez avec:")
    logger.info("    python main.py data/raw/")


if __name__ == "__main__":
    main()
