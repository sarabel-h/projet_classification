import sys
import os
import cv2
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gabarits.detecteur_cnie import DetecteurCNIE

print("🔍 DÉBOGAGE DÉTECTEUR CNIE")
print("=" * 50)

# Initialisation du détecteur
detecteur = DetecteurCNIE(
    chemin_gabarits="models/gabarits/gabarits_maroc.json"
)

# Image de test
image_path = "data/raw/carte_identite/cnie01.png"

if not Path(image_path).exists():
    print(f"❌ Image non trouvée: {image_path}")
    exit(1)

print(f"\n📸 Image: {image_path}")

# Charger l'image
img = cv2.imread(image_path)
h, w = img.shape[:2]

print(f"   Dimensions: {w}x{h} pixels")
print(f"   Ratio: {w/h:.3f}")

# Bande haute (info debug)
bande = img[:int(h * 0.18), :]
print(f"   Couleur bande haut (moyenne BGR): {bande.mean(axis=(0,1))}")

# Analyse
resultat = detecteur.analyser_image(image_path)

print(f"\n📊 RÉSULTAT COMPLET:")
print(f"   Est CNIE: {resultat['est_cnie']}")
print(f"   Score: {resultat['score']}/100")
print(f"   Ratio calculé: {resultat['ratio']}")
print(f"   Message: {resultat['message']}")

# Diagnostic correct (FIX IMPORTANT)
print("\n🧠 DIAGNOSTIC")
if not resultat["est_cnie"]:
    print("❓ POURQUOI PAS DÉTECTÉ ?")
    print("   - Score insuffisant")
    print("   - Causes possibles :")
    print("     1. OCR faible (image floue / sombre)")
    print("     2. CIN non détecté")
    print("     3. Bande haute peu colorée")
    print("     4. Ratio trop éloigné")
else:
    print("✅ CNIE correctement détectée")

print("\n✔️ Test terminé")