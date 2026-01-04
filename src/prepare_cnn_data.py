import os, shutil
from pdf2image import convert_from_path

RAW_DIR = 'data/raw'
OUT_DIR = 'data/cnn_images'
IMG_EXTS = ('.jpg', '.jpeg', '.png')

def prepare_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    for label in os.listdir(RAW_DIR):
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        out_label_dir = os.path.join(OUT_DIR, label)
        os.makedirs(out_label_dir, exist_ok=True)
        for file in os.listdir(label_dir):
            src_path = os.path.join(label_dir, file)
            if file.lower().endswith(IMG_EXTS):
                shutil.copy(src_path, out_label_dir)
            elif file.lower().endswith('.pdf'):
                pages = convert_from_path(src_path)
                for i, page in enumerate(pages):
                    img_name = f"{os.path.splitext(file)[0]}_p{i}.jpg"
                    page.save(os.path.join(out_label_dir, img_name), 'JPEG')
    print('Prepared CNN images in', OUT_DIR)

if __name__ == '__main__':
    prepare_dataset()
