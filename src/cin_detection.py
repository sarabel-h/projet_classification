
import cv2
import numpy as np

def analyze_cin_structure(image_path_or_array):
    """
    Analyse spécifique pour distinguer Recto/Verso CIN Marocaine.
    Retourne: "recto", "verso" ou "unknown" avec un score.
    """
    try:
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
        else:
            img = np.array(image_path_or_array)
            # Si image PIL
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1] # RGB to BGR

        if img is None: return None, 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Détection MRZ (Machine Readable Zone) -> Typique du Verso
        # Le MRZ est une zone de texte dense en bas, très contrastée
        # On utilise un gradient morphologique (Blackhat) pour isoler le texte sombre sur fond clair
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # On cherche des lignes horizontales fortes en bas de l'image
        h, w = gray.shape
        bottom_part = blackhat[int(h*0.6):, :] # 40% bas
        
        # Projection horizontale
        proj = np.sum(bottom_part, axis=1)
        # S'il y a des pics forts réguliers, c'est probablement un MRZ
        peaks = np.sum(proj > (np.max(proj) * 0.4))
        
        is_verso = peaks > 15 # Seuil heuristique
        
        if is_verso:
            return "verso", 0.90
            
        # 2. Détection Visage -> Typique du Recto
        # Utilisation simple Haar Cascade (très léger)
        # Note: En prod, utiliser un modèle DL face detection serait mieux
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return "recto", 0.85
            
        # Par défaut, si pas de MRZ et pas de visage détecté, c'est ambigu
        return "recto", 0.51 # Léger biais recto par défaut pour les CIN

    except Exception as e:
        print(f"Erreur CIN analysis: {e}")
        return "unknown", 0.0
