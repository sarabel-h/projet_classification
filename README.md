# Document Classifier (NLP + CV) - Offline
Project skeleton: multimodal classifier combining OCR, CamemBERT (FR), and CNN vision.
Data should be added under `data/raw/<class_name>/`.

Run sequence (example):
1. Prepare CNN images: python src/prepare_cnn_data.py
2. Train CNN: python src/train_cnn.py
3. Train CamemBERT: python src/train_camembert.py
4. Run classification: python main.py
5. Evaluate: python src/evaluate_all.py
