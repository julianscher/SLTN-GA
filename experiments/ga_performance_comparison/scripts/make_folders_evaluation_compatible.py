import os

from utilities.helper_functions import get_results_path


def rename_folders(directory):
    # Get all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Sort the folders to maintain order (optional, depending on OS behavior)
    folders.sort()

    # Rename folders from job0 to job<N-1>
    for i, folder in enumerate(folders):
        new_name = f"job_{i}"
        old_path = os.path.join(directory, folder)
        new_path = os.path.join(directory, new_name)

        if old_path != new_path:  # Avoid renaming if the name is already correct
            os.rename(old_path, new_path)
            print(f"Renamed: {folder} -> {new_name}")
        else:
            print(f"Skipping: {folder} (already correctly named)")


if __name__ == "__main__":
    directory = f"{get_results_path()}/ga_performance_results/make_blobs_normalized/make_two_blobs"
    if os.path.exists(directory) and os.path.isdir(directory):
        rename_folders(directory)
    else:
        print("Invalid directory path.")
