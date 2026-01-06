import os
import json

# Kiểm tra folder patient_records
records_dir = "patient_records"
if os.path.exists(records_dir):
    print(f"Found {len(os.listdir(records_dir))} patient folders")
    for folder in os.listdir(records_dir)[:2]:  # Check first 2
        folder_path = os.path.join(records_dir, folder)
        record_file = os.path.join(folder_path, "record.json")
        if os.path.exists(record_file):
            with open(record_file, 'r', encoding='utf-8') as f:
                record = json.load(f)
            print(f"Folder: {folder}")
            print(f"Record ID: {record['record_id']}")
            print(f"Images in record: {list(record['images'].keys())}")

            # Check if image files exist
            for img_type, img_path in record['images'].items():
                exists = os.path.exists(img_path)
                print(f"  {img_type}: {img_path} -> {'EXISTS' if exists else 'MISSING'}")
            print()
else:
    print("patient_records folder not found")