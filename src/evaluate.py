import os, json
from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = 'results'
RAW_DIR = 'data/raw'

y_true = []
y_pred = []
unknown_count = 0
total = 0

for label in os.listdir(RAW_DIR):
    label_dir = os.path.join(RAW_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        result_file = os.path.join(RESULTS_DIR, file + '.json')
        if not os.path.exists(result_file):
            continue
        with open(result_file, encoding='utf-8') as f:
            pred = json.load(f)['predicted_class']
        total += 1
        if pred == 'unknown':
            unknown_count += 1
            continue
        y_true.append(label)
        y_pred.append(pred)

print('=== Rapport de classification (hors unknown) ===')
print(classification_report(y_true, y_pred, zero_division=0))
print('=== Matrice de confusion ===')
print(confusion_matrix(y_true, y_pred))
print('\n=== Rejet (unknown) ===')
print(f'Documents rejetés : {unknown_count}/{total} ({round(unknown_count/total*100,2)}%)')
