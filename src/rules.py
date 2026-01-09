# Dictionnaire de mots-clés pondérés (Français + Arabe)
# Structure: "mot": poids (float)

KEYWORDS_DB = {
    "facture_electricite": {
        # Français
        "electricite": 2, "kwh": 4, "basse tension": 2, "puissance": 1, 
        "redal": 1, "lydec": 1, "amendis": 1, "onee": 1, "consommation": 1,
        # Arabe
        "الكهرباء": 3, "فاتورة الكهرباء": 4, "استهلاك": 1, "جهد": 1
    },
    "facture_eau": {
        # Français
        "eau": 2, "m3": 4, "assainissement": 3, "tranche": 1, "index": 1, 
        "potable": 2, "metre cube": 3, "agence": 1,
        # Arabe
        "الماء": 3, "فاتورة الماء": 4, "التطهير": 3, "متر مكعب": 3, "الوكالة": 1
    },
    "releve_bancaire": {
        # Français
        "solde": 2, "credit": 1, "debit": 1, "rib": 3, "banque": 2, 
        "date operation": 2, "date valeur": 2, "virement": 1,
        # Arabe
        "كشف حساب": 4, "رصيد": 2, "دائن": 1, "مدين": 1, "وكالة": 1, "البنك": 2
    },
    "piece_identite": {
        # Français
        "carte nationale": 3, "identite": 2, "royaume du maroc": 2, 
        "ne le": 1, "valable": 1, "cin": 2, "verso": 1, "recto": 1,
        # Arabe
        "المملكة المغربية": 3, "البطاقة الوطنية": 4, "التعريف": 2, "صلاحية": 1
    },
    "document_employeur": {
        # Français
        "salaire": 3, "paie": 2, "bulletin": 2, "cnss": 2, 
        "net a payer": 3, "employeur": 2, "contrat": 1, "travail": 1,
        # Arabe
        "راتب": 3, "أجر": 2, "شهادة عمل": 4, "الصندوق الوطني": 2
    }
}

# --- RÈGLES D'EXCLUSION (VETO) ---
# Si ces mots sont présents, on pénalise la classe.
EXCLUSIONS = {
    "facture_eau": ["kwh", "tension", "electrique", "الكهرباء"],
    "facture_electricite": ["assainissement", "potable", "التطهير"],
    "piece_identite": ["solde", "kwh", "m3", "salaire", "رصيد"],
    "releve_bancaire": ["kwh", "m3", "carte nationale", "الكهرباء"],
    "document_employeur": ["kwh", "m3", "solde"]
}

def get_rule_scores(text):
    if not text:
        return {k: 0.0 for k in KEYWORDS_DB.keys()}, "unknown"
        
    text_lower = text.lower()
    scores = {k: 0.0 for k in KEYWORDS_DB.keys()}
    
    # 1. Scoring positif
    for cls, kw_dict in KEYWORDS_DB.items():
        for word, weight in kw_dict.items():
            if word in text_lower:
                scores[cls] += weight
                
    # 2. Scoring négatif (Pénalités)
    for cls, veto_words in EXCLUSIONS.items():
        for veto in veto_words:
            if veto in text_lower:
                scores[cls] -= 10.0 
                
    # 3. Normalisation
    total = sum(max(0, s) for s in scores.values())
    if total == 0:
        return {k: 0.0 for k in scores}, "unknown"
        
    normalized = {k: max(0, v) / total for k, v in scores.items()}
    best_cls = max(normalized, key=normalized.get)
    
    return normalized, best_cls