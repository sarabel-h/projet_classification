# src/offline_manager.py
import torch
import json
from pathlib import Path

class GestionnaireOffline:
    def __init__(self):
        self.base_dir = Path(".")
        self.cache = {}
    
    def get_cv_model(self, nom="resnet50"):
        """Charge ResNet50 ou MobileNetV2"""
        if f"cv_{nom}" in self.cache:
            print(f"📦 Utilisation cache: {nom}")
            return self.cache[f"cv_{nom}"]
        
        chemin = self.base_dir / "models" / "cv" / f"{nom}.pth"
        
        if not chemin.exists():
            print(f"❌ Modèle {nom} non trouvé")
            print("   Exécutez: python setup_offline.py")
            return None
        
        print(f"🔄 Chargement {nom}...")
        
        if nom == "resnet50":
            from torchvision import models
            model = models.resnet50(pretrained=False)
        elif nom == "mobilenet_v2":
            from torchvision import models
            model = models.mobilenet_v2(pretrained=False)
        else:
            print(f"❌ Modèle inconnu: {nom}")
            return None
        
        model.load_state_dict(torch.load(chemin, map_location="cpu"))
        model.eval()
        
        self.cache[f"cv_{nom}"] = model
        print(f"✅ {nom} chargé")
        return model
    
    def get_nlp_model(self):
        """Charge CamemBERT"""
        if "nlp_camembert" in self.cache:
            print("📦 Utilisation cache: CamemBERT")
            return self.cache["nlp_camembert"]
        
        chemin = self.base_dir / "models" / "nlp" / "camembert"
        
        if not chemin.exists():
            print("❌ CamemBERT non trouvé")
            print("   Exécutez: python setup_offline.py")
            return None, None
        
        print("🔄 Chargement CamemBERT...")
        
        try:
            from transformers import CamembertModel, CamembertTokenizer
            tokenizer = CamembertTokenizer.from_pretrained(str(chemin))
            model = CamembertModel.from_pretrained(str(chemin))
        except ImportError:
            print("❌ Transformers non installé")
            print("   pip install transformers")
            return None, None
        
        result = (model, tokenizer)
        self.cache["nlp_camembert"] = result
        print("✅ CamemBERT chargé")
        return result
    
    def get_gabarits(self):
        """Charge la config des gabarits marocains"""
        chemin = self.base_dir / "models" / "gabarits" / "gabarits_maroc.json"
        
        if not chemin.exists():
            print("❌ Gabarits non trouvés")
            return {}
        
        with open(chemin, "r", encoding="utf-8") as f:
            gabarits = json.load(f)
        
        print(f"✅ Gabarits chargés ({len(gabarits)} classes)")
        return gabarits
    
    def get_info(self):
        """Affiche les infos du système"""
        info = {
            "modeles_cv": [],
            "modeles_nlp": [],
            "gabarits": False,
            "ocr": False
        }
        
        # Vérifier CV
        cv_dir = self.base_dir / "models" / "cv"
        if cv_dir.exists():
            for f in cv_dir.glob("*.pth"):
                info["modeles_cv"].append(f.stem)
        
        # Vérifier NLP
        nlp_dir = self.base_dir / "models" / "nlp"
        if nlp_dir.exists():
            for d in nlp_dir.iterdir():
                if d.is_dir():
                    info["modeles_nlp"].append(d.name)
        
        # Vérifier gabarits
        gabarits_file = self.base_dir / "models" / "gabarits" / "gabarits_maroc.json"
        info["gabarits"] = gabarits_file.exists()
        
        # Vérifier OCR
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            info["ocr"] = True
        except:
            info["ocr"] = False
        
        return info

def tester():
    """Teste le gestionnaire"""
    print("🧪 TEST OFFLINE MANAGER")
    print("=" * 40)
    
    manager = GestionnaireOffline()
    
    # Test CV
    print("\n1. Test modèles CV:")
    model_cv = manager.get_cv_model("resnet50")
    if model_cv:
        print("   ✅ ResNet50 OK")
    
    # Test NLP
    print("\n2. Test modèle NLP:")
    model_nlp, tokenizer = manager.get_nlp_model()
    if model_nlp:
        print("   ✅ CamemBERT OK")
    
    # Test gabarits
    print("\n3. Test gabarits:")
    gabarits = manager.get_gabarits()
    if gabarits:
        print(f"   ✅ {len(gabarits)} classes de documents")
        for classe in gabarits:
            print(f"      - {classe}")
    
    # Infos système
    print("\n4. Infos système:")
    info = manager.get_info()
    print(f"   Modèles CV: {info['modeles_cv']}")
    print(f"   Modèles NLP: {info['modeles_nlp']}")
    print(f"   Gabarits: {'✅' if info['gabarits'] else '❌'}")
    print(f"   OCR: {'✅' if info['ocr'] else '❌'}")
    
    print("\n" + "=" * 40)
    print("✅ Test terminé!")

if __name__ == "__main__":
    tester()