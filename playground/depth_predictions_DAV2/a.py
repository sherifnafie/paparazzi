import os

def add_prefix_to_files(directory='.', prefix='depth_', exclude_files=['a.py']):
    """
    Add a prefix to all filenames in a directory except for specified files.
    
    Args:
        directory: Directory to process (default: current directory)
        prefix: Prefix to add to filenames
        exclude_files: List of filenames to exclude from renaming
    """
    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Process each file
    for filename in files:
        # Skip excluded files
        if filename in exclude_files:
            print(f"Skipping excluded file: {filename}")
            continue
        
        # Skip files that already have the prefix
        if filename.startswith(prefix):
            print(f"Skipping already prefixed file: {filename}")
            continue
        
        # Create new filename with prefix
        new_filename = prefix + filename
        
        # Construct full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    # Run the function to rename files in the current directory
    add_prefix_to_files()
    print("Renaming complete.")