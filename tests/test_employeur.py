# tests/test_employeur.py
import sys
import os
from pathlib import Path

# ==============================================================================
# 1. CORRECTION DES CHEMINS
# ==============================================================================
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
try:
    from src.gabarits.detecteur_employeur import DetecteurEmployeur, pdf_vers_image_opencv
    print("✅ Modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    sys.exit(1)

# ==============================================================================
# 3. CONFIGURATION
# ==============================================================================
CHEMIN_JSON = project_root / "models" / "gabarits" / "gabarits_maroc.json"
DOSSIER_DATA = project_root / "data" / "raw" / "document_employeur"

def main():
    print("🔍 DÉTECTEUR DOCUMENTS EMPLOYEUR - VERSION AMÉLIORÉE")
    print("=" * 60)
    print(f"📂 Racine du projet : {project_root}")
    print(f"📄 Fichier gabarits : {CHEMIN_JSON}")
    print(f"📂 Dossier données  : {DOSSIER_DATA}")
    print("=" * 60)

    # 1. Vérification fichiers
    if not CHEMIN_JSON.exists():
        print(f"\n❌ Fichier JSON introuvable: {CHEMIN_JSON}")
        return
    
    if not DOSSIER_DATA.exists():
        print(f"\n❌ Dossier données introuvable: {DOSSIER_DATA}")
        return

    # 2. Instanciation détecteur
    detecteur = DetecteurEmployeur(str(CHEMIN_JSON))
    
    # 3. Lister les fichiers PDF
    fichiers_pdf = list(DOSSIER_DATA.glob("*.pdf")) + list(DOSSIER_DATA.glob("*.PDF"))
    
    if not fichiers_pdf:
        print(f"⚠️  Aucun PDF trouvé dans {DOSSIER_DATA}")
        print(f"   Formats supportés: PDF")
        return
    
    print(f"\n📁 {len(fichiers_pdf)} fichiers PDF trouvés")
    print("=" * 60)

    # 4. Traitement des fichiers
    resultats = []
    
    for i, chemin_pdf in enumerate(fichiers_pdf, 1):
        print(f"\n[{i}/{len(fichiers_pdf)}] 📄 Analyse de: {chemin_pdf.name}")
        print("-" * 40)
        
        # Conversion PDF -> Image
        img = pdf_vers_image_opencv(chemin_pdf)
        
        if img is None:
            print("   ❌ Erreur de conversion PDF")
            continue
        
        # Analyse
        resultat = detecteur.analyser(img)
        
        # Afficher résultats
        if resultat["valide"]:
            print(f"   ✅ VALIDE - {resultat['type_document']}")
            print(f"   📊 Score: {resultat['score']}/100")
            
            # Détails
            details = resultat.get("details", {})
            if details:
                print(f"\n   📋 DÉTAILS:")
                for key, value in details.items():
                    if key == "secteur":
                        print(f"      🏢 Secteur: {value}")
                    elif key == "cnss":
                        print(f"      🔢 CNSS: {value}")
                    elif key == "salaire_detecte":
                        print(f"      💰 Salaire: {value}")
                    elif key == "employeur":
                        print(f"      🏭 Employeur: {value}")
                    elif key == "date":
                        print(f"      📅 Date: {value}")
                    elif key == "signature":
                        print(f"      ✍️  Signature: Oui")
                    else:
                        print(f"      • {key}: {value}")
        else:
            print(f"   ❌ NON VALIDE - {resultat['type_document']}")
            print(f"   📊 Score: {resultat['score']}/100 (minimum: 50)")
            if "erreur" in resultat:
                print(f"   ⚠️  Erreur: {resultat['erreur']}")
        
        # Stocker pour synthèse
        resultats.append((chemin_pdf.name, resultat))
        
        # Aperçu texte (optionnel)
        if "texte_analyse" in resultat and resultat['score'] < 60:
            print(f"\n   🔤 Aperçu texte (pour debug):")
            print(f"      {resultat['texte_analyse'][:200]}...")

    # 5. Synthèse finale
    print(f"\n" + "=" * 60)
    print("📈 SYNTHÈSE FINALE")
    print("=" * 60)
    
    if resultats:
        detectes = sum(1 for _, r in resultats if r["valide"])
        total = len(resultats)
        
        print(f"\n📊 STATISTIQUES:")
        print(f"   Documents analysés: {total}")
        print(f"   Documents valides: {detectes}")
        print(f"   Taux de succès: {detectes/total*100:.1f}%" if total > 0 else "N/A")
        
        # Distribution par type
        types_comptes = {}
        for _, r in resultats:
            if r["valide"]:
                type_doc = r["type_document"]
                types_comptes[type_doc] = types_comptes.get(type_doc, 0) + 1
        
        print(f"\n🏷️  RÉPARTITION:")
        for type_doc, compte in types_comptes.items():
            print(f"   • {type_doc}: {compte}")
        
        print(f"\n📋 RÉSULTATS DÉTAILLÉS:")
        for nom_fichier, resultat in resultats:
            statut = "✅ VALIDE" if resultat["valide"] else "❌ INVALIDE"
            print(f"   • {nom_fichier}: {statut} ({resultat['score']}/100) - {resultat['type_document']}")
    else:
        print(f"❌ Aucun résultat à afficher")

    print(f"\n" + "=" * 60)
    print("✅ TEST TERMINÉ")
    print("=" * 60)

if __name__ == "__main__":
    main()