import sys
import os
import cv2
from pathlib import Path

# Ajout du chemin racine pour trouver les modules src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gabarits.detecteur_cnie import DetecteurCNIE

print("🔍 TEST CNIE MULTI-VERSIONS (DEBUG AVANCÉ)")
print("=" * 60)

# Initialisation
detecteur = DetecteurCNIE()

# Dossier des images
dossier_img = Path("data/raw/carte_identite")

if not dossier_img.exists():
    print(f"❌ Le dossier {dossier_img} n'existe pas !")
    sys.exit(1)

fichiers = list(dossier_img.glob("*"))
print(f"📂 {len(fichiers)} fichiers trouvés.\n")

for img_path in fichiers:
    # On ne traite que les images
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: 
        continue
    
    print(f"📸 Analyse : {img_path.name}")
    
    # Lancement de l'analyse
    res = detecteur.analyser_image(img_path)
    
    # Affichage du statut
    statut = "✅" if res["est_cnie"] else "❌"
    color_type = "\033[92m" if res["est_cnie"] else "\033[91m" # Vert ou Rouge pour le terminal
    reset = "\033[0m"
    
    print(f"   {statut} Type : {color_type}{res['type']}{reset} (Score: {res['score']})")
    
    # Affichage des détails positifs
    if res["est_cnie"]:
        if res.get('cin_detecte'):
            print(f"      🆔 Numéro CIN : {res['cin_detecte']}")
        print(f"      📝 Preuves : {', '.join(res['details'])}")
    
    # Affichage systématique du Debug (très important pour toi)
    print(f"      🎨 Couleurs : {res.get('debug_couleur')}")
    print(f"      📖 Texte lu (début) : \"{res.get('debug_texte')}\"")
    
    print("-" * 40)