import pandas as pd
import os
import shutil

# Load CSV
csv_path = r"C:\Users\khali\OneDrive\Desktop\Image Processing\covid-chestxray-dataset-master\covid-chestxray-dataset-master\metadata.csv"
df = pd.read_csv(csv_path)

# Set paths
image_folder = r"C:\Users\khali\OneDrive\Desktop\Image Processing\covid-chestxray-dataset-master\covid-chestxray-dataset-master\images"
destination_folder = r"C:\Users\khali\OneDrive\Desktop\Image Processing\train\NO_FINDINGS"
os.makedirs(destination_folder, exist_ok=True)

# Keyword to look for in the finding column
keyword = "No Finding"

# Loop through rows
for index, row in df.iterrows():
    finding = str(row['finding']).lower()

    if keyword in finding:
        filename = row['filename']
        source_path = os.path.join(image_folder, filename)
        dest_path = os.path.join(destination_folder, filename)

        if os.path.isfile(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved: {filename}")
        else:
            print(f"File not found: {filename}")

