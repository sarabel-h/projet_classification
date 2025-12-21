# Projet de Classification de Documents Administratifs - Version Offline

## 📋 Vue d'ensemble

Pipeline intelligent de classification automatique de documents administratifs en 5 catégories principales, fonctionnant complètement offline sans connexion internet.

**Catégories supportées:**
- 🆔 Pièce d'identité (CNIE - Recto et Verso)
- 🏦 Relevé bancaire (Différentes Banques)
- ⚡ Facture d'électricité (Différentes Régies)
- 💧 Facture d'eau (Différentes Régies)
- 💼 Document employeur (Bulletins de paie + Attestations de Travail)

## 🏗️ Architecture du Système

```
project_classification/
├── models/                          # Modèles pré-entraînés
│   ├── cv/                          # Modèles Computer Vision (ResNet50, EfficientNet)
│   ├── nlp/                         # Modèles NLP (CamemBERT, embeddings)
│   └── gabarits/                    # Configuration des gabarits
│
├── data/                            # Données du projet
│   ├── raw/                         # Documents bruts (à mettre vos images ici!)
│   ├── processed/                   # Données prétraitées
│   └── annotations/                 # Labels et métadonnées
│
├── src/                             # Code source
│   ├── preprocessing/               # Prétraitement (PDF → images → texte)
│   ├── computer_vision/             # Classification par vision
│   ├── nlp/                         # Traitement du langage naturel
│   ├── fusion/                      # Fusion multimodale
│   ├── gabarits/                    # Détection de structures
│   ├── utils/                       # Utilitaires
│   └── offline_manager.py           # Gestion des modèles offline
│
├── tests/                           # Tests unitaires
├── logs/                            # Logs d'exécution
├── output/                          # Résultats de classification
│
├── main.py                          # Point d'entrée principal
├── setup_offline.py                 # Script d'initialisation
├── config.json                      # Configuration du projet
├── requirements.txt                 # Dépendances Python
└── README.md                        # Ce fichier
```

## 🚀 Installation et Setup

### 1. **Prérequis système**

```bash
# Python 3.8+
python --version

# Tesseract OCR (système)
# Linux: sudo apt-get install tesseract-ocr tesseract-ocr-fra
# macOS: brew install tesseract
# Windows: Télécharger https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. **Installation des dépendances Python**

```bash
cd project_classification
pip install -r requirements.txt
```

### 3. **Initialisation du projet**

```bash
python setup_offline.py
```

Cela va:
- ✓ Créer la structure de répertoires
- ✓ Vérifier les dépendances
- ✓ Configurer Tesseract OCR
- ✓ Créer le fichier config.json

## 📁 Où ajouter VOS données?

### **Pour les images d'entraînement/test:**

```
data/raw/
├── piece_identite/
│   ├── doc_1.jpg
│   ├── doc_2.jpg
│   └── ...
├── releve_bancaire/
│   ├── doc_1.jpg
│   └── ...
├── facture_electricite/
│   ├── doc_1.jpg
│   └── ...
├── facture_eau/
│   ├── doc_1.jpg
│   └── ...
└── document_employeur/
    ├── doc_1.jpg
    └── ...
```

### **Pour les fichiers PDF:**

```
data/raw/
└── mixed_documents/
    ├── batch_1.pdf
    ├── batch_2.pdf
    └── ...
```

## 🔧 Utilisation

### **Mode simple - Classifier un document**

```bash
python main.py data/raw/mon_document.pdf
```

### **Mode batch - Classifier un répertoire**

```bash
python main.py data/raw/
```

### **Avec configuration personnalisée**

```bash
python main.py data/raw/ --config custom_config.json --output my_output/
```

## 📊 Architecture du Pipeline

```
Document (PDF/Image)
    ↓
[1] Conversion PDF → Images
    ↓
[2] Extraction features gabarits (structure, ratio, texte, etc.)
    ↓
[3] Prétraitement CV (redimensionnement, normalisation)
    ↓
[4] Classification Computer Vision (ResNet50)
    ├── Images + Features gabarits → Modèle hybride
    └── Output: predictions CV + confiance
    ↓
[5] Extraction OCR + Prétraitement texte
    ├── Text extraction (Tesseract)
    └── Cleaning et normalization
    ↓
[6] Classification NLP (CamemBERT)
    ├── Embeddings + Motifs sémantiques
    └── Output: predictions NLP + confiance
    ↓
[7] FUSION MULTIMODALE
    ├── Accord CV+NLP → Accepter
    ├── CV confiant + Gabarits valides → Favoriser CV
    ├── NLP confiant + Motifs textuels → Favoriser NLP
    ├── Fusion pondérée → Score final
    └── Output: Classe finale + Confiance
    ↓
Classification finale + Confiance + Rapport détaillé
```

## 🎯 Modules détaillés

### **Module 1: Configuration Offline**
- Gestion complète des modèles locaux
- Cache en mémoire pour performances
- Vérification d'intégrité des modèles
- Support A/B testing

### **Module 2: Système de Gabarits**
- Détection de structures tabulaires (Hough)
- Détection de zones de photo (Cascade classifiers)
- Calcul de densité de texte
- Analyse de ratio d'aspect
- Détection de zones de signature

### **Module 3: Computer Vision Hybride**
- ResNet50 comme backbone
- Features structurelles (gabarits) fusionnées
- Fine-tuning sur données administratives
- Évaluation détaillée par classe

### **Module 4: NLP Offline**
- 3 approches complémentaires:
  1. **Motifs sémantiques**: Mots-clés spécifiques par classe
  2. **CamemBERT**: Modèle transformers français
  3. **RNN+Embeddings**: Séquences avec attention
- Extraction OCR avec Tesseract
- Correction post-OCR par dictionnaires

### **Module 5: Fusion Multimodale**
- Stratégies intelligentes de fusion
- Règles métier par classe
- Système de confiance robuste
- Bac "À vérifier" pour cas limites

### **Module 6: Pipeline Principal**
- Intégration complète
- Gestion d'erreurs
- Interface CLI + Web
- Logging détaillé
- Performance optimisée

## 📈 Métriques et Évaluation

Le système mesure:

**Par classe:**
- Accuracy, Precision, Recall, F1-score
- Matrice de confusion
- Analyse des erreurs

**Globale:**
- Accuracy globale (objectif: ≥90%)
- Temps de traitement par document
- Taux de rejet (documents à vérifier)

**Robustesse:**
- Performance sur documents de mauvaise qualité
- Consistance sur différentes régies
- Stabilité sur plusieurs exécutions

## 🛠️ Fichiers importants à connaître

| Fichier | Description |
|---------|-------------|
| `main.py` | Point d'entrée principal du pipeline |
| `setup_offline.py` | Script d'initialisation |
| `config.json` | Configuration complète du projet |
| `src/offline_manager.py` | Gestionnaire de modèles offline |
| `src/preprocessing/preprocessor.py` | Prétraitement documents |
| `src/gabarits/detector.py` | Détection de gabarits |
| `src/fusion/multimodal_fusion.py` | Fusion CV+NLP |

## 🔍 Dépannage

### Erreur: "pytesseract not found"
```bash
# Linux
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract

# Windows: Télécharger depuis GitHub
```

### Erreur: "No module named..."
```bash
pip install -r requirements.txt
```

### Erreur: "PDF conversion failed"
```bash
# Vérifier que pdf2image fonctionne
pip install pdf2image
# Sur Linux, installer poppler
sudo apt-get install poppler-utils
```

## 📚 Ressources recommandées

- **Computer Vision**: OpenCV docs, PyTorch docs
- **NLP**: Hugging Face Transformers, CamemBERT
- **OCR**: Tesseract documentation
- **Architecture**: Clean Code principles, modular design

## 🤝 Contribution et Amélioration

Points possibles d'amélioration:
1. Intégrer de vrais modèles pré-entraînés (ResNet50, CamemBERT)
2. Créer un dataset d'entraînement annoté
3. Fine-tuner les modèles
4. Interface web avancée
5. Monitoring en temps réel
6. Support multi-langue

## 📝 Notes importantes

- **Offline**: Tous les modèles fonctionnent localement, zéro dépendance internet
- **Modulaire**: Chaque module peut être testé indépendamment
- **Extensible**: Facile d'ajouter de nouvelles classes ou modèles
- **Production-ready**: Logging, error handling, monitoring inclus
- **Benchmarking**: Scripts de comparaison des modèles inclus

## ⚠️ À faire IMPÉRATIVEMENT

1. **Télécharger les modèles pré-entraînés:**
   - ResNet50 depuis PyTorch/TorchVision
   - CamemBERT depuis Hugging Face
   - Placer dans `models/cv/`, `models/nlp/`

2. **Ajouter vos données:**
   - Créer des sous-dossiers dans `data/raw/`
   - Par classe de document
   - Voir structure ci-dessus

3. **Valider Tesseract:**
   ```bash
   python -c "import pytesseract; pytesseract.get_tesseract_version()"
   ```

## 📞 Support

Pour les problèmes:
1. Vérifier les logs dans `logs/`
2. Consulter le README détaillé de chaque module
3. Tester chaque module indépendamment

Bon projet! 🚀
