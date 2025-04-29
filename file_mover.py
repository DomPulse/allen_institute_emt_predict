import os
import shutil

# Settings
source_directory = r'F:\Big_MET_data\morpho_trans'      # Folder where your messy folders are
destination_directory = r'F:\Big_MET_data\just_trans_extracted'   # Folder where you want all files collected

# Create destination if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Walk through all subdirectories
for root, dirs, files in os.walk(source_directory):
    for file in files:
        source_path = os.path.join(root, file)
        destination_path = os.path.join(destination_directory, file)

        # If a file with the same name already exists, rename it
        if os.path.exists(destination_path):
            base, ext = os.path.splitext(file)
            counter = 1
            while True:
                new_filename = f"{base}_{counter}{ext}"
                destination_path = os.path.join(destination_directory, new_filename)
                if not os.path.exists(destination_path):
                    break
                counter += 1

        print(f"Moving {source_path} -> {destination_path}")
        shutil.move(source_path, destination_path)

print("✓ Done moving all files!")
