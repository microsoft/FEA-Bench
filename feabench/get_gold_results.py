import argparse
from datasets import load_dataset, load_from_disk
import json
import os

def main(dataset_path, split, save_dir, file_name):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)[split]
    else:
        dataset = load_dataset(dataset_path)[split]

    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in dataset:
            save_dict = {
                "instance_id": item["instance_id"],
                "model_name_or_path": "gold",
                "full_output": "",
                "model_patch": item["patch"]
            }
            f.write(json.dumps(save_dict, ensure_ascii=False) + "\n")
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save dataset to JSONL file.")
    parser.add_argument("--dataset_name_or_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output file.")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the output file.")
    
    args = parser.parse_args()
    
    main(args.dataset_name_or_path, "test", args.save_dir, args.file_name)
