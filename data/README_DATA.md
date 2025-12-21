# 📁 Guide d'organisation des données

## Structure recommandée

```
data/
├── raw/                             # Données brutes
│   ├── piece_identite/             # Images de cartes d'identité
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   │
│   ├── releve_bancaire/            # Relevés bancaires
│   │   ├── banque_cih_001.jpg
│   │   ├── maroc_telecom_001.jpg
│   │   └── ...
│   │
│   ├── facture_electricite/        # Factures d'électricité
│   │   ├── one_001.jpg
│   │   ├── redal_001.jpg
│   │   └── ...
│   │
│   ├── facture_eau/                # Factures d'eau
│   │   ├── lydec_001.jpg
│   │   ├── redal_eau_001.jpg
│   │   └── ...
│   │
│   └── document_employeur/         # Documents employeur
│       ├── bulletin_001.jpg
│       ├── attestation_001.jpg
│       └── ...
│
├── processed/                       # Données prétraitées
│   ├── images_normalized/
│   ├── text_extracted/
│   └── features/
│
└── annotations/                    # Labels et métadonnées
    ├── labels.csv
    ├── train_split.txt
    ├── val_split.txt
    └── test_split.txt
```

## Format des données

### Images
- **Formats acceptés**: JPG, PNG, GIF
- **Résolution recommandée**: 224x224 minimum pour CV, 300+ DPI pour OCR
- **Format PDF**: Automatiquement converti en images

### CSV des labels
```csv
filename,class,quality,source,notes
image_001.jpg,piece_identite,high,scanned,Clear photo
image_002.jpg,releve_bancaire,medium,mobile,Some noise
```

### Splits (train/val/test)
```
image_001.jpg
image_002.jpg
image_003.jpg
...
```

## Format de fichiers

### Images de Pièce d'Identité
- Recto et verso séparés ou combinés
- Photo claire obligatoire
- Informations textuelles visibles
- **Exemple**: `CIN_RECTO_20240115_001.jpg`

### Relevés Bancaires
- Toutes les banques marocaines acceptées
- Structure tabulaire avec soldes et opérations
- Au moins 3 mois de relevé recommandé
- **Exemple**: `RELEVE_BNP_202401_001.jpg`

### Factures Électricité
- ONE, REDAL, LYDEC acceptés
- Tableau de consommation en kWh
- Période de facturation claire
- **Exemple**: `FACTURE_ONE_202401_001.jpg`

### Factures Eau
- LYDEC, REDAL, ONE acceptés
- Tableau de consommation en m³
- Index de consommation visible
- **Exemple**: `FACTURE_EAU_LYDEC_202401_001.jpg`

### Documents Employeur
- Bulletins de paie avec détails salaires
- Attestations de travail
- En-tête entreprise visible
- **Exemple**: `BULLETIN_MAROC_TELECOM_202401_001.pdf`

## Naming convention recommandée

```
{TYPE}_{PROVIDER}_{DATE}_{INDEX}.{EXT}

Exemples:
- CIN_MOROCCO_20240115_001.jpg
- RELEVE_BNP_202401_001.jpg
- FACTURE_ONE_ELECTRICITE_202401_001.jpg
- FACTURE_LYDEC_EAU_202401_001.jpg
- BULLETIN_OCP_202401_001.pdf
```

## Qualité des données

### Hautement recommandé:
- ✅ Images claires et bien éclairées
- ✅ Documents entiers dans le cadre
- ✅ Peu de rotation ou perspective distorsion
- ✅ Résolution suffisante (≥300 DPI pour OCR)
- ✅ Pas de documents endommagés

### À éviter:
- ❌ Images floues ou pixellisées
- ❌ Rotation excessive (>15°)
- ✘ Documents partiels ou coupés
- ❌ Mauvaise luminosité
- ❌ Surchargement d'images

## Volume de données recommandé

Pour un bon entraînement:

| Classe | Minimum | Idéal | Stretch |
|--------|---------|-------|---------|
| Pièce d'identité | 50 | 200+ | 500+ |
| Relevé bancaire | 50 | 200+ | 500+ |
| Facture électricité | 50 | 200+ | 500+ |
| Facture eau | 50 | 200+ | 500+ |
| Document employeur | 50 | 200+ | 500+ |
| **TOTAL** | **250** | **1000+** | **2500+** |

## Étapes à suivre

### 1. Créer la structure
```bash
mkdir -p data/raw/{piece_identite,releve_bancaire,facture_electricite,facture_eau,document_employeur}
mkdir -p data/processed data/annotations
```

### 2. Ajouter vos images
```bash
# Exemple pour les pièces d'identité
cp /path/to/cin_images/* data/raw/piece_identite/

# Exemple pour les relevés
cp /path/to/bank_statements/* data/raw/releve_bancaire/
```

### 3. Créer les annotations
```bash
# data/annotations/labels.csv
filename,class,quality,source
CIN_001.jpg,piece_identite,high,scanned
RELEVE_001.jpg,releve_bancaire,medium,mobile
```

### 4. Créer les splits
```bash
# data/annotations/train_split.txt
CIN_001.jpg
CIN_002.jpg
RELEVE_001.jpg
...

# data/annotations/val_split.txt
CIN_050.jpg
RELEVE_050.jpg
...

# data/annotations/test_split.txt
CIN_100.jpg
RELEVE_100.jpg
...
```

## Scripts d'aide (à créer)

### Script de vérification des données
```python
# Vérifier que toutes les images existent et sont valides
python scripts/validate_data.py --data_dir data/raw/
```

### Script d'organisation
```python
# Organiser les images par classe
python scripts/organize_data.py --source /path/to/images --dest data/raw/
```

## Conseils pratiques

1. **Commencez petit**: 20-30 images par classe pour tester
2. **Augmentez progressivement**: Ajoutez plus de données au fil du temps
3. **Variez les sources**: Différentes régies, banques, qualités
4. **Documentez**: Notez la source, la date, les particularités
5. **Nettoyez**: Supprimez les images corrompues ou mal étiquetées
6. **Sauvegardez**: Backups réguliers de vos données

## Format d'annotation avancée

Pour plus de précision, créer un `annotations/metadata.json`:

```json
{
  "images": [
    {
      "filename": "CIN_001.jpg",
      "class": "piece_identite",
      "quality": "high",
      "source": "scanned",
      "provider": "morocco",
      "date_scanned": "2024-01-15",
      "dimensions": [1024, 640],
      "dpi": 300,
      "notes": "Clear recto, visible security features"
    },
    {
      "filename": "RELEVE_001.jpg",
      "class": "releve_bancaire",
      "quality": "medium",
      "source": "mobile",
      "provider": "bmce",
      "date_document": "2024-01",
      "dimensions": [1080, 1920],
      "text_quality": "medium",
      "notes": "Slight tilt, 2 pages"
    }
  ]
}
```

Bonne organisation! 📊
