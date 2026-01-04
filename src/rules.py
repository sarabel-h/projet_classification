KEYWORDS = {
    "piece_identite": [
        "carte nationale",
        "royaume du maroc",
        "etat civil",
        "valable jusqu'au",
        "المملكة المغربية",
        "البطاقة الوطنية للتعريف",
        "رقم الحالة المدنية",
        "صالحة الى غاية"
    ],

    "facture_electricite": [
        "electricite",
        "redevance fixe electricite",
        "kwh",
        "facture electricite",
        "consommation electricite",
        "الكهرباء",
        "فاتورة الكهرباء"
    ],

    "facture_eau": [
        "eau",
        "assainissement",
        "redevance fixe eau",
        "eau et assainissement",
        "m3",
        "facture eau",
        "consommation eau",
        "الماء",
        "التطهير",
        "فاتورة الماء"
    ],

    "releve_bancaire": [
        "releve de compte",
        "banque",
        "solde",
        "compte",
        "debit",
        "credit",
        "agence",
        "valeur",
        "rib",
        "كشف حساب",
        "رصيد",
        "دائنية",
        "مدينة",
        "وكالة"
    ],

    "document_employeur": [
        "attestation",
        "soussigne",
        "salaire",
        "droit",
        "mensuel",
        "travail",
        "attestons",
        "net",
        "brut"
    ]
}

def classify_text(text):
    text = text.lower() if text else ""
    scores = {}
    for label, words in KEYWORDS.items():
        scores[label] = sum(word in text for word in words)
    max_score = max(scores.values()) if scores else 0
    if max_score == 0:
        return "unknown", 0.0
    # pick first best label (simple, deterministic)
    best_label = max(scores, key=scores.get)
    confidence = max_score / sum(scores.values()) if sum(scores.values())>0 else 0.0
    return best_label, round(confidence, 2)
