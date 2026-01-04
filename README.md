# Offline Document Classifier
Simple offline application to classify documents into:
- piece_identite
- releve_bancaire
- facture_electricite
- facture_eau
- document_employeur

Pipeline:
PDF/Image → OCR → Keywords rules → CV fallback → Final class
