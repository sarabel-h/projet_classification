import sys
import os
import cv2
from pathlib import Path

# Imports dynamiques
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.gabarits.detecteur_factures import DetecteurFacture, pdf_vers_images
except ImportError:
    print("❌ Erreur d'importation.")
    sys.exit(1)

BASE_DIR = Path(parent_dir)
# IMPORTANT: Le chemin vers ton nouveau JSON complet
CHEMIN_JSON = BASE_DIR / "models" / "gabarits" / "gabarits_maroc.json" 
DOSSIER_DATA = BASE_DIR / "data" / "raw"

def main():
    print("🔍 TEST CLASSIFICATION FACTURES (Piloté par JSON)")
    print(f"📄 Config : {CHEMIN_JSON}")
    print("=" * 60)

    # Initialisation avec le JSON
    detecteur = DetecteurFacture(str(CHEMIN_JSON))

    fichiers_cibles = [
        "facture_eau2.jpg",
        "facture-electricite-eau-amandis2.pdf",
        "facture-electricite-eau-redal1.pdf",
        "facture-electricite-eau-lydec2.pdf",
        "facture-electricite-eau-redeema2.pdf",
        "facture-electricite-lydec1.pdf",
        "facture-electricite-eau-redal2.pdf"
    ]
    
    fichiers_trouves = []
    for fichier in fichiers_cibles:
        trouve = list(DOSSIER_DATA.rglob(fichier))
        if trouve: fichiers_trouves.append(trouve[0])

    print(f"\n📁 {len(fichiers_trouves)} fichiers trouvés.\n")

    for chemin in fichiers_trouves:
        print(f"📄 Document : {chemin.name}")
        
        images = []
        if chemin.suffix.lower() == ".pdf":
            images = pdf_vers_images(chemin)
        else:
            img = cv2.imread(str(chemin))
            if img is not None: images = [img]

        if images:
            res = detecteur.analyser(images)
            
            if res["valide"]:
                type_f = res['type_facture']
                icon = "💧⚡" if "Mixte" in type_f else ("⚡" if "Électricité" in type_f else ("💧" if "Eau" in type_f else "❓"))
                
                print(f"   ✅ {icon} Type : \033[1m{type_f}\033[0m")
                print(f"      🏢 Fournisseur : {res['fournisseur']}")
                print(f"      💰 Montant     : {res['montant_total_prob']} DH")
                print(f"      🎯 Score       : {res['score']}/100")
            else:
                print(f"   ❌ {res['message']}")
        else:
            print("   ❌ Erreur technique")
            
        print("-" * 40)

if __name__ == "__main__":
    main()