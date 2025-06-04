import os

# Set the path to the directory containing the files
directory = r"C:\Users\khali\OneDrive\Desktop\Image Processing\RandomvsChestModel\Random"
# Loop through all files in the directory
for filename in os.listdir(directory):
    # Get the full path of the file
    file_path = os.path.join(directory, filename)
    
    # Check if it is a file (not a folder)
    if os.path.isfile(file_path):
        # Check the file extension
        if not filename.lower().endswith(('.jpeg', '.jpg')):
            print(f"Removing: {filename}")