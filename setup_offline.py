import os
import torch
import json
from pathlib import Path

print("=" * 60)
print("🚀 SETUP OFFLINE - DOCUMENTS MAROCAINS")
print("=" * 60)
import os
import torch
import json
from pathlib import Path

print("=" * 60)
print("🚀 SETUP OFFLINE - DOCUMENTS MAROCAINS")
print("=" * 60)

# 1. Créer la structure des dossiers
print("\n📁 Création de la structure...")

dossiers = [
    "models/cv",
    "models/nlp",
    "models/gabarits",
    "models/ocr",
    "data/raw/carte_identite",
    "data/raw/releve_bancaire",
    "data/raw/facture_electricite",
    "data/raw/facture_eau",
    "data/raw/document_employeur",
    "data/processed",
    "data/annotations",
    "src/preprocessing",
    "src/computer_vision",
    "src/nlp",
    "src/fusion",
    "src/gabarits",
    "src/utils",
    "tests"
]

for dossier in dossiers:
    Path(dossier).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dossier}")

# Créer __init__.py dans src
for sous_dossier in ["preprocessing", "computer_vision", "nlp", "fusion", "gabarits", "utils"]:
    (Path("src") / sous_dossier / "__init__.py").touch(exist_ok=True)

# 2. Télécharger ResNet50
print("\n🖼️  Téléchargement ResNet50...")
try:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    torch.save(model.state_dict(), "models/cv/resnet50.pth")
    print("  ✓ ResNet50 sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur ResNet50: {e}")

# 3. Télécharger MobileNetV2 (modèle léger)
print("\n📱 Téléchargement MobileNetV2...")
try:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    torch.save(model.state_dict(), "models/cv/mobilenet_v2.pth")
    print("  ✓ MobileNetV2 sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur MobileNetV2: {e}")

# 4. Télécharger CamemBERT
print("\n🇫🇷 Téléchargement CamemBERT...")
try:
    from transformers import CamembertModel, CamembertTokenizer
    
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    
    # Sauvegarder
    model_dir = "models/nlp/camembert"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print("  ✓ CamemBERT sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur CamemBERT: {e}")
    print("  Astuce: pip install transformers")

# 5. Créer les gabarits marocains
print("\n🎯 Création des gabarits marocains...")

gabarits_maroc = {
  "carte_identite": {
    "description": "CNIE biométrique marocaine - Double ligne bilingue",
    "structure_bande_rouge": {
      "lignes": 2,
      "ligne1": {
        "segments": [
          {
            "position": "gauche",
            "texte": "ROYAUME DU MAROC",
            "langue": "fr"
          },
          {
            "position": "centre",
            "type": "motif_ornemental"
          },
          {
            "position": "droite",
            "texte": "المملكة المغربية",
            "langue": "ar"
          }
        ]
      },
      "ligne2": {
        "segments": [
          {
            "position": "gauche",
            "texte": "carte nationale d'identité",
            "langue": "fr"
          },
          {
            "position": "droite",
            "texte": "البطاقة الوطنية للتعريف",
            "langue": "ar"
          }
        ]
      },
      "couleur_fond": "#CC0000",
      "couleur_texte": "#000000"
    },
    "features": [
      {
        "nom": "bande_rouge_haut",
        "type": "couleur",
        "zone": [
          0,
          0,
          1,
          0.15
        ]
      },
      {
        "nom": "motif_centre",
        "type": "pattern",
        "zone": [
          0.35,
          0.02,
          0.65,
          0.13
        ]
      },
      {
        "nom": "drapeau_bas",
        "type": "couleur",
        "zone": [
          0.1,
          0.85,
          0.25,
          0.95
        ]
      },
      {
        "nom": "format_carte",
        "type": "ratio",
        "valeur": 1.586
      }
    ]
  },
  "releve_bancaire_maroc": {
    "description": "Relevés bancaires multibanques (CIH, Attijariwafa, BP, Barid Bank)",
    "features": [
      {
        "nom": "entete_banque_logo",
        "type": "region",
        "description": "Logo de la banque et agence (Haut de page)",
        "zone": [
          0.0,
          0.0,
          1.0,
          0.20
        ]
      },
      {
        "nom": "info_client_rib",
        "type": "region",
        "description": "Zone contenant le nom du client, l'adresse et le RIB (souvent un tableau ou ligne)",
        "zone": [
          0.0,
          0.15,
          1.0,
          0.38
        ]
      },
      {
        "nom": "tableau_operations",
        "type": "region",
        "description": "Le corps principal contenant la liste des transactions (Date, Valeur, Débit, Crédit)",
        "zone": [
          0.02,
          0.35,
          0.98,
          0.85
        ]
      },
      {
        "nom": "pied_page_soldes",
        "type": "region",
        "description": "Bas de page contenant souvent le Nouveau Solde ou les totaux",
        "zone": [
          0.0,
          0.80,
          1.0,
          1.0
        ]
      },
      {
        "nom": "detection_rib",
        "type": "regex",
        "description": "Détection automatique des 24 chiffres du RIB marocain",
        "patterns": [
          "\\d{3}\\s*\\d{3}\\s*\\d{12,16}\\s*\\d{2}",
          "RIB\\s*[:.]?\\s*\\d+"
        ]
      },
      {
        "nom": "detection_dates",
        "type": "regex",
        "description": "Détection des formats de date (JJ/MM/AAAA ou JJ/MM/AA)",
        "patterns": [
          "\\d{2}/\\d{2}/\\d{4}",
          "\\d{2}/\\d{2}/\\d{2}"
        ]
      },
      {
        "nom": "mots_cles_solde",
        "type": "regex",
        "description": "Repère les lignes de solde (début ou fin)",
        "patterns": [
          "SOLDE",
          "NOUVEAU SOLDE",
          "ANCIEN SOLDE",
          "TOTAL"
        ]
      }
    ]
  },
  "facture_eau_electricite": {
    "description": "Factures Eau & Electricité (Logique Robuste V4)",
    "categories": {
      "electricite": {
        "keywords": [
          "electricite", "electrique", "eiectricite", "flectricite", 
          "energie active", "energie reactive",
          "moyenne tension", "basse tension", "mt/bt",
          "eclairage", "puissance", 
          "audiovisuel", "csave", "bav", "prom. paysage",
          "redevance fixe", "prime fixe", "entretien compteur",
          "كهرباء", "kahraba"
        ],
        "units_regex": [
          "k\\s*[w|v]\\s*h", 
          "kilowatt",
          "kwh"
        ]
      },
      "eau": {
        "keywords": [
          "eau", "assainissement", "tranche eau", "potable",
          "debit", "consommation eau", "pollution",
          "redevance fixe assainissement", "redevance fixe eau",
          "entretien compteur eau",
          "ماء", "تطهير", "shourb"
        ],
        "units_regex": [
          "\\d+\\s*m3", 
          "metre cube", 
          "metres cubes"
        ]
      }
    },
    "providers": {
      "REDAL": ["redal", "ريضال", "rabat", "sale", "skhirat", "temara"],
      "LYDEC": ["lydec", "lyonnaise", "ليدك", "casablanca", "mohammedia"],
      "AMENDIS": ["amendis", "amanidis", "أمانديس", "tanger", "tetouan"],
      "RADEEMA": ["radeema", "radeem", "راديما", "marrakech"],
      "ONE": ["office national", "onee", "onel"]
    },
    "general_keywords": [
      "facture", "montant", "total", "consommation", "dh", "dhs", 
      "فاتورة", "مبلغ", "واجب", "payer", "net a payer"
    ]
  },
  "attestation_employeur": {
    "description": "Attestations Travail & Salaire (Secteur Privé & Public)",
    "features": [
      {
        "nom": "zone_titre",
        "type": "region",
        "description": "Haut de page pour identifier le type de document",
        "zone": [0.0, 0.0, 1.0, 0.30]
      },
      {
        "nom": "corps_texte",
        "type": "region",
        "description": "Le texte principal qui contient le nom et la fonction",
        "zone": [0.0, 0.20, 1.0, 0.70]
      },
      {
        "nom": "bas_page_signature",
        "type": "region",
        "description": "Zone de cachet et signature",
        "zone": [0.0, 0.60, 1.0, 1.0]
      },
      {
        "nom": "mots_cles_type",
        "type": "keywords",
        "patterns": {
          "salaire": ["attestation de salaire", "etat de salaire", "émoluments", "salary"],
          "travail": ["attestation de travail", "certificat de travail", "travaille en qualité", "est employé"]
        }
      },
      {
        "nom": "regex_cnss",
        "type": "regex",
        "description": "Numéro d'immatriculation CNSS (souvent 9 chiffres)",
        "patterns": [
          "CNSS\\s*[:.]?\\s*\\d{9}",
          "immatriculé.*\\d{9}",
          "n°\\s*\\d{9}"
        ]
      },
      {
        "nom": "regex_montants",
        "type": "regex",
        "description": "Détection de salaire (ex: 5000 DH)",
        "patterns": [
          "\\d+[\\s\\.]?\\d{2,3}[,.]\\d{2}\\s*(DH|MAD|DIRHAMS)",
          "salaire.*\\d+"
        ]
      }
    ]
  }
}
with open("models/gabarits/gabarits_maroc.json", "w", encoding="utf-8") as f:
    json.dump(gabarits_maroc, f, indent=2, ensure_ascii=False)
print("  ✓ Gabarits marocains créés")

# 6. Créer config.json
print("\n  Création de config.json...")

config = {
    "projet": "Classification Documents Marocains",
    "version": "1.0",
    "classes": ["carte_identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"],
    "image_size": 224,
    "langue_ocr": "fra"
}

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print("  ✓ config.json créé")

# 7. Créer OCR config
print("\n🔤 Création config OCR...")

ocr_config = {
    "langue": "fra",
    "config": "--oem 3 --psm 3",
    "preprocessing": {"dpi": 300}
}

with open("models/ocr/config.json", "w", encoding="utf-8") as f:
    json.dump(ocr_config, f, indent=2, ensure_ascii=False)
print("  ✓ Config OCR créée")

print("\n" + "=" * 60)
print("✅ SETUP TERMINÉ AVEC SUCCÈS !")
print("=" * 60)
print("\n📦 Modèles dans: models/")
print("🎯 Gabarits: models/gabarits/gabarits_maroc.json")
print("\n🔄 Testez avec: python src/offline_manager.py")
# 1. Créer la structure des dossiers
print("\n📁 Création de la structure...")

dossiers = [
    "models/cv",
    "models/nlp",
    "models/gabarits",
    "models/ocr",
    "data/raw/carte_identite",
    "data/raw/releve_bancaire",
    "data/raw/facture_electricite",
    "data/raw/facture_eau",
    "data/raw/document_employeur",
    "data/processed",
    "data/annotations",
    "src/preprocessing",
    "src/computer_vision",
    "src/nlp",
    "src/fusion",
    "src/gabarits",
    "src/utils",
    "tests"
]

for dossier in dossiers:
    Path(dossier).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {dossier}")

# Créer __init__.py dans src
for sous_dossier in ["preprocessing", "computer_vision", "nlp", "fusion", "gabarits", "utils"]:
    (Path("src") / sous_dossier / "__init__.py").touch(exist_ok=True)

# 2. Télécharger ResNet50
print("\n🖼️  Téléchargement ResNet50...")
try:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    torch.save(model.state_dict(), "models/cv/resnet50.pth")
    print("  ✓ ResNet50 sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur ResNet50: {e}")

# 3. Télécharger MobileNetV2 (modèle léger)
print("\n📱 Téléchargement MobileNetV2...")
try:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    torch.save(model.state_dict(), "models/cv/mobilenet_v2.pth")
    print("  ✓ MobileNetV2 sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur MobileNetV2: {e}")

# 4. Télécharger CamemBERT
print("\n🇫🇷 Téléchargement CamemBERT...")
try:
    from transformers import CamembertModel, CamembertTokenizer
    
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    
    # Sauvegarder
    model_dir = "models/nlp/camembert"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print("  ✓ CamemBERT sauvegardé")
except Exception as e:
    print(f"  ✗ Erreur CamemBERT: {e}")
    print("  Astuce: pip install transformers")

# 5. Créer les gabarits marocains
print("\n🎯 Création des gabarits marocains...")

gabarits_maroc = {
  "carte_identite": {
    "description": "CNIE biométrique marocaine - Double ligne bilingue",
    "structure_bande_rouge": {
      "lignes": 2,
      "ligne1": {
        "segments": [
          {
            "position": "gauche",
            "texte": "ROYAUME DU MAROC",
            "langue": "fr"
          },
          {
            "position": "centre",
            "type": "motif_ornemental"
          },
          {
            "position": "droite",
            "texte": "المملكة المغربية",
            "langue": "ar"
          }
        ]
      },
      "ligne2": {
        "segments": [
          {
            "position": "gauche",
            "texte": "carte nationale d'identité",
            "langue": "fr"
          },
          {
            "position": "droite",
            "texte": "البطاقة الوطنية للتعريف",
            "langue": "ar"
          }
        ]
      },
      "couleur_fond": "#CC0000",
      "couleur_texte": "#000000"
    },
    "features": [
      {
        "nom": "bande_rouge_haut",
        "type": "couleur",
        "zone": [
          0,
          0,
          1,
          0.15
        ]
      },
      {
        "nom": "motif_centre",
        "type": "pattern",
        "zone": [
          0.35,
          0.02,
          0.65,
          0.13
        ]
      },
      {
        "nom": "drapeau_bas",
        "type": "couleur",
        "zone": [
          0.1,
          0.85,
          0.25,
          0.95
        ]
      },
      {
        "nom": "format_carte",
        "type": "ratio",
        "valeur": 1.586
      }
    ]
  },
  "releve_bancaire_maroc": {
    "description": "Relevés bancaires multibanques (CIH, Attijariwafa, BP, Barid Bank)",
    "features": [
      {
        "nom": "entete_banque_logo",
        "type": "region",
        "description": "Logo de la banque et agence (Haut de page)",
        "zone": [
          0.0,
          0.0,
          1.0,
          0.20
        ]
      },
      {
        "nom": "info_client_rib",
        "type": "region",
        "description": "Zone contenant le nom du client, l'adresse et le RIB (souvent un tableau ou ligne)",
        "zone": [
          0.0,
          0.15,
          1.0,
          0.38
        ]
      },
      {
        "nom": "tableau_operations",
        "type": "region",
        "description": "Le corps principal contenant la liste des transactions (Date, Valeur, Débit, Crédit)",
        "zone": [
          0.02,
          0.35,
          0.98,
          0.85
        ]
      },
      {
        "nom": "pied_page_soldes",
        "type": "region",
        "description": "Bas de page contenant souvent le Nouveau Solde ou les totaux",
        "zone": [
          0.0,
          0.80,
          1.0,
          1.0
        ]
      },
      {
        "nom": "detection_rib",
        "type": "regex",
        "description": "Détection automatique des 24 chiffres du RIB marocain",
        "patterns": [
          "\\d{3}\\s*\\d{3}\\s*\\d{12,16}\\s*\\d{2}",
          "RIB\\s*[:.]?\\s*\\d+"
        ]
      },
      {
        "nom": "detection_dates",
        "type": "regex",
        "description": "Détection des formats de date (JJ/MM/AAAA ou JJ/MM/AA)",
        "patterns": [
          "\\d{2}/\\d{2}/\\d{4}",
          "\\d{2}/\\d{2}/\\d{2}"
        ]
      },
      {
        "nom": "mots_cles_solde",
        "type": "regex",
        "description": "Repère les lignes de solde (début ou fin)",
        "patterns": [
          "SOLDE",
          "NOUVEAU SOLDE",
          "ANCIEN SOLDE",
          "TOTAL"
        ]
      }
    ]
  },
  "facture_eau_electricite": {
    "description": "Factures Eau & Electricité (Logique Robuste V4)",
    "categories": {
      "electricite": {
        "keywords": [
          "electricite", "electrique", "eiectricite", "flectricite", 
          "energie active", "energie reactive",
          "moyenne tension", "basse tension", "mt/bt",
          "eclairage", "puissance", 
          "audiovisuel", "csave", "bav", "prom. paysage",
          "redevance fixe", "prime fixe", "entretien compteur",
          "كهرباء", "kahraba"
        ],
        "units_regex": [
          "k\\s*[w|v]\\s*h", 
          "kilowatt",
          "kwh"
        ]
      },
      "eau": {
        "keywords": [
          "eau", "assainissement", "tranche eau", "potable",
          "debit", "consommation eau", "pollution",
          "redevance fixe assainissement", "redevance fixe eau",
          "entretien compteur eau",
          "ماء", "تطهير", "shourb"
        ],
        "units_regex": [
          "\\d+\\s*m3", 
          "metre cube", 
          "metres cubes"
        ]
      }
    },
    "providers": {
      "REDAL": ["redal", "ريضال", "rabat", "sale", "skhirat", "temara"],
      "LYDEC": ["lydec", "lyonnaise", "ليدك", "casablanca", "mohammedia"],
      "AMENDIS": ["amendis", "amanidis", "أمانديس", "tanger", "tetouan"],
      "RADEEMA": ["radeema", "radeem", "راديما", "marrakech"],
      "ONE": ["office national", "onee", "onel"]
    },
    "general_keywords": [
      "facture", "montant", "total", "consommation", "dh", "dhs", 
      "فاتورة", "مبلغ", "واجب", "payer", "net a payer"
    ]
  },
  "attestation_employeur": {
    "description": "Attestations Travail & Salaire (Secteur Privé & Public)",
    "features": [
      {
        "nom": "zone_titre",
        "type": "region",
        "description": "Haut de page pour identifier le type de document",
        "zone": [0.0, 0.0, 1.0, 0.30]
      },
      {
        "nom": "corps_texte",
        "type": "region",
        "description": "Le texte principal qui contient le nom et la fonction",
        "zone": [0.0, 0.20, 1.0, 0.70]
      },
      {
        "nom": "bas_page_signature",
        "type": "region",
        "description": "Zone de cachet et signature",
        "zone": [0.0, 0.60, 1.0, 1.0]
      },
      {
        "nom": "mots_cles_type",
        "type": "keywords",
        "patterns": {
          "salaire": ["attestation de salaire", "etat de salaire", "émoluments", "salary"],
          "travail": ["attestation de travail", "certificat de travail", "travaille en qualité", "est employé"]
        }
      },
      {
        "nom": "regex_cnss",
        "type": "regex",
        "description": "Numéro d'immatriculation CNSS (souvent 9 chiffres)",
        "patterns": [
          "CNSS\\s*[:.]?\\s*\\d{9}",
          "immatriculé.*\\d{9}",
          "n°\\s*\\d{9}"
        ]
      },
      {
        "nom": "regex_montants",
        "type": "regex",
        "description": "Détection de salaire (ex: 5000 DH)",
        "patterns": [
          "\\d+[\\s\\.]?\\d{2,3}[,.]\\d{2}\\s*(DH|MAD|DIRHAMS)",
          "salaire.*\\d+"
        ]
      }
    ]
  }
}
with open("models/gabarits/gabarits_maroc.json", "w", encoding="utf-8") as f:
    json.dump(gabarits_maroc, f, indent=2, ensure_ascii=False)
print("  ✓ Gabarits marocains créés")

# 6. Créer config.json
print("\n  Création de config.json...")

config = {
    "projet": "Classification Documents Marocains",
    "version": "1.0",
    "classes": ["carte_identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"],
    "image_size": 224,
    "langue_ocr": "fra"
}

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print("  ✓ config.json créé")

# 7. Créer OCR config
print("\n🔤 Création config OCR...")

ocr_config = {
    "langue": "fra",
    "config": "--oem 3 --psm 3",
    "preprocessing": {"dpi": 300}
}

with open("models/ocr/config.json", "w", encoding="utf-8") as f:
    json.dump(ocr_config, f, indent=2, ensure_ascii=False)
print("  ✓ Config OCR créée")

print("\n" + "=" * 60)
print("✅ SETUP TERMINÉ AVEC SUCCÈS !")
print("=" * 60)
print("\n📦 Modèles dans: models/")
print("🎯 Gabarits: models/gabarits/gabarits_maroc.json")
print("\n🔄 Testez avec: python src/offline_manager.py")