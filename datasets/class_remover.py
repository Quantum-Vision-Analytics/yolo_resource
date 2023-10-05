import os
import shutil
from datetime import datetime

cur_dir = os.getcwd()
target_folder = 'labels'
current_time = datetime.now().strftime("_%H%M%S-%d%m%Y")

# Search for all subfolders that has the target folder name
def search_folders(working_dir):
    found_dirs = []
    items = os.listdir(working_dir)
    
    if not items:
        return None

    for item in items :
        if item == target_folder:
           found_dirs.append(os.path.join(working_dir, target_folder)) 
        else:
            item_path = os.path.join(working_dir, item)

            if os.path.isdir(item_path):
                # Recursively search for "labels" folders in the subdirectory
                sub_dirs = search_folders(item_path)
                
                # If a target folder is found in the subdirectory, add it to the list
                if sub_dirs and target_folder in sub_dirs:
                    found_dirs.append(os.path.join(item_path, target_folder))
                
                # If the subdirectory has other subdirectories, add them as well
                if sub_dirs:
                    found_dirs.extend(sub_dirs)
    
    return found_dirs

print("This script removes all the unnecessary classes from label files of the working directory and subfolders and reorders remaning ones."
+ "\nThe script only works for YOLO formatted labels with \'.txt\' file extension.")
print("Input the indexes of the classes you want to remain\nCorrect format : 4 6 9 2 5")

# Getting correct class indexes from the user
while True:
    inp = input().split(" ")
    if all(index.isdigit() for index in inp):
        tmp = []
        for val in inp:
            if val in tmp:
                print("Please input a sequence of different indexes.")
                break
            else:
                tmp.append(val)

        if len(inp) == len(tmp):
            class_indexes = inp
            del tmp
            break
    else:
        print("Please input integers that are not less than 0")

class_dict = {val : str(x) for x, val in enumerate(class_indexes)}
print(f"Class conversion: {class_dict}")

# Call the function to find "labels" folders
label_folders = search_folders(cur_dir)

# Adjusting every file in the found folders
if label_folders:
    print("Folders found:")

    for folder in label_folders:
        print(folder)
        files = os.listdir(folder)
        if not files:
            continue
        
        # Copying the old label files with the current time stamp
        shutil.copytree(folder,folder + current_time)

        for file in files:
            if file.endswith("txt"):
                with open(os.path.join(folder, file), "r+") as f:
                    # Reading all the info of a file to make adjustments
                    lines = f.readlines()
                    f.seek(0)

                    # Rewriting filtered and reordered data to the same file
                    for line in lines:
                        data = line.split(" ")
                        if data and data[0] in class_dict:
                            data[0] = class_dict[data[0]]
                            f.write(" ".join(data))

                    # Truncate any remaining content if the new content is shorter
                    f.truncate()
else:
    print("\"labels\" folder is not found in the current and sub folders. Relocate and run the script again.")

input("Runtime ended without problems, press any key to close this window.")
