import sys
import os
from pathlib import Path

# On récupère le chemin absolu du dossier actuel (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# On récupère le dossier parent (projet-classification/)
parent_dir = os.path.dirname(current_dir)
# On ajoute la racine du projet au "path" de Python pour qu'il trouve 'src'
sys.path.append(parent_dir)
from src.gabarits.detecteur_releves import DetecteurReleveBancaire, pdf_vers_image_opencv

# ==============================================================================
# On utilise Path pour gérer les chemins proprement (Mac/Windows/Linux)
BASE_DIR = Path(parent_dir) 
CHEMIN_JSON = BASE_DIR / "models" / "gabarits" / "gabarits_maroc.json"
DOSSIER_DATA = BASE_DIR / "data" / "raw" / "releve_bancaire"

def main():
    print("🔍 DÉMARRAGE DU TEST (Portable - Windows/Mac/Linux)")
    print("=" * 60)
    print(f"📂 Dossier de travail : {BASE_DIR}")

    # 1. Vérification des fichiers
    if not DOSSIER_DATA.exists():
        print(f"❌ Le dossier {DOSSIER_DATA} n'existe pas.")
        print("   Créez-le et mettez vos PDFs dedans.")
        # Pour le debug, on affiche où il cherche
        print(f"   Chemin cherché : {DOSSIER_DATA}")
        return

    fichiers = list(DOSSIER_DATA.glob("*.pdf"))
    if not fichiers:
        print(f"❌ Aucun fichier PDF trouvé dans {DOSSIER_DATA}")
        return

    # 2. Instanciation du détecteur
    # On passe le chemin absolu du JSON pour éviter les erreurs
    detecteur = DetecteurReleveBancaire(str(CHEMIN_JSON))
    
    succes = 0
    total = len(fichiers)

    print(f"📁 {total} fichiers trouvés à analyser...\n")

    for fichier in fichiers:
        print(f"📄 Analyse de : {fichier.name}")
        
        # Conversion (Universelle)
        img = pdf_vers_image_opencv(fichier)

        if img is not None:
            # Détection
            resultat = detecteur.analyser(img)
            
            if resultat["valide"]:
                succes += 1
                d = resultat["details"]
                print(f"   ✅ \033[92mVALIDE\033[0m (Score: {resultat['score']})")
                print(f"      🏦 Banque : {d.get('banque', 'Inconnue')}")
                print(f"      🔢 RIB    : {d.get('rib', 'Non trouvé')}")
            else:
                print(f"   ❌ \033[91mINVALID\033[0m (Score: {resultat['score']})")
        else:
            print("   ⚠️ Erreur : Impossible de convertir ce PDF.")
            
        print("-" * 40)

    print("\n" + "=" * 60)
    print(f"📊 RÉSULTAT FINAL : {succes}/{total} documents reconnus.")

if __name__ == "__main__":
    main()