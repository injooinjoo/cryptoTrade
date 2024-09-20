import os
import hashlib


def calculate_hash(file_path, block_size=65536):
    hash_algorithm = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b""):
            hash_algorithm.update(block)
    return hash_algorithm.hexdigest()


def find_duplicate_files(folder_path):
    hashes = {}
    duplicates = []

    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            file_hash = calculate_hash(file_path)

            if file_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[file_hash] = file_path

    return duplicates


def delete_files(file_paths):
    for file_path in file_paths:
        # os.remove(file_path)
        print(f"Deleted: {file_path}")


# 사용 예시
folder_path = r'C:\Users\injoo\dwhelper'
duplicates = find_duplicate_files(folder_path)

if duplicates:
    delete_files(duplicates)
else:
    print("No duplicates found.")
