"""
Gestionnaire de modèles offline pour l'initialisation complète sans internet
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OfflineModelManager:
    """Gère le chargement et la cache des modèles localement"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialise le gestionnaire de modèles offline
        
        Args:
            models_dir: Chemin vers le répertoire contenant les modèles
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.models_manifest = self.models_dir / "manifest.json"
        self._load_manifest()
    
    def _load_manifest(self):
        """Charge le manifeste des modèles disponibles"""
        if self.models_manifest.exists():
            with open(self.models_manifest, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "cv": {},
                "nlp": {},
                "gabarits": {}
            }
    
    def _save_manifest(self):
        """Sauvegarde le manifeste des modèles"""
        with open(self.models_manifest, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def verify_model_integrity(self, model_path: str, expected_hash: Optional[str] = None) -> bool:
        """
        Vérifie l'intégrité d'un modèle sauvegardé
        
        Args:
            model_path: Chemin vers le modèle
            expected_hash: Hash MD5 attendu (optionnel)
            
        Returns:
            True si le modèle est valide
        """
        model_file = self.models_dir / model_path
        if not model_file.exists():
            logger.warning(f"Modèle non trouvé: {model_path}")
            return False
        
        # Calcul du hash
        hash_md5 = hashlib.md5()
        with open(model_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        actual_hash = hash_md5.hexdigest()
        
        if expected_hash and actual_hash != expected_hash:
            logger.error(f"Hash mismatch pour {model_path}")
            return False
        
        logger.info(f"Modèle vérifié: {model_path} (hash: {actual_hash})")
        return True
    
    def get_model(self, model_type: str, model_name: str):
        """
        Charge un modèle depuis le cache ou le disque
        
        Args:
            model_type: Type de modèle ('cv', 'nlp', 'gabarits')
            model_name: Nom du modèle
            
        Returns:
            Le modèle chargé
        """
        cache_key = f"{model_type}_{model_name}"
        
        # Vérifier le cache en mémoire
        if cache_key in self.cache:
            logger.info(f"Modèle chargé depuis le cache: {cache_key}")
            return self.cache[cache_key]
        
        # Charger depuis le disque
        model_path = self.models_dir / model_type / f"{model_name}.pth"
        if not model_path.exists():
            logger.error(f"Modèle non trouvé: {model_path}")
            return None
        
        logger.info(f"Chargement du modèle: {model_path}")
        # Implémentation réelle dépendra du framework utilisé
        # Pour l'instant, c'est un placeholder
        
        self.cache[cache_key] = model_path
        return model_path
    
    def list_available_models(self) -> Dict:
        """Liste tous les modèles disponibles"""
        models = {
            "cv": list((self.models_dir / "cv").glob("*.pth")) if (self.models_dir / "cv").exists() else [],
            "nlp": list((self.models_dir / "nlp").glob("*.pth")) if (self.models_dir / "nlp").exists() else [],
            "gabarits": list((self.models_dir / "gabarits").glob("*.json")) if (self.models_dir / "gabarits").exists() else []
        }
        return {k: [str(p.stem) for p in v] for k, v in models.items()}
    
    def get_model_info(self, model_type: str, model_name: str) -> Optional[Dict]:
        """Récupère les informations d'un modèle"""
        if model_type in self.manifest and model_name in self.manifest[model_type]:
            return self.manifest[model_type][model_name]
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = OfflineModelManager("models")
    print("Modèles disponibles:")
    print(manager.list_available_models())
