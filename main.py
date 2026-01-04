from src.pipeline import process_folder

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "results"
    process_folder(input_dir, output_dir)
    print("Classification terminée.")
