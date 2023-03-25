import os
import uuid
import time
from shutil import copyfile, copy


def copy_file(from_dir, dest_dir):
    dir_count = 0
    files = [(sub_dir, file_name) for sub_dir, dir_names, file_names in os.walk(from_dir) for file_name in file_names if
             os.path.splitext(file_name)[-1] == '.mp4']
    for sub_dir, file_name in files:
        new_file_name = sub_dir + '\\' + str(uuid.uuid4())[:8] + file_name
        file_path = sub_dir + '\\' + file_name
        os.rename(file_path, new_file_name)

        dir_count += 1
        print('\r', dir_count, '/', len(files), new_file_name, '====>', dest_dir, end='', flush=True)
        copy(new_file_name, dest_dir)
        os.remove(new_file_name)
    # print(os.path.splitext(file_names))


def copy_file_max(from_dir, dest_dir):
    dir_count = 0
    files = [(sub_dir, file_name) for sub_dir, dir_names, file_names in os.walk(from_dir) for file_name in file_names if
             os.path.splitext(file_name)[-1] == '.MP4']
    for sub_dir, file_name in files:
        new_file_name = sub_dir + '\\' + str(uuid.uuid4())[:8] + file_name
        file_path = sub_dir + '\\' + file_name
        os.rename(file_path, new_file_name)

        dir_count += 1
        print('\r', dir_count, '/', len(files), new_file_name, '====>', dest_dir, end='', flush=True)
        copy(new_file_name, dest_dir)
        os.remove(new_file_name)
    # print(os.path.splitext(file_names))


def copy_file_all(from_dir, dest_dir):
    dir_count = 0
    files = [(sub_dir, file_name, os.path.splitext(file_name)[-1]) for sub_dir, dir_names, file_names in
             os.walk(from_dir) for file_name in file_names]
    for sub_dir, file_name, ext in files:
        new_file_name = sub_dir + '\\' + str(uuid.uuid4())[:8] + file_name
        file_path = sub_dir + '\\' + file_name
        os.rename(file_path, new_file_name)

        dir_count += 1
        # print('\r', dir_count, '/', len(files), new_file_name, '====>', dest_dir, end='', flush=True)
        # time.sleep(0.1)
        print(ext)
        # copy(new_file_name, dest_dir)
        # os.remove(new_file_name)
    # print(os.path.splitext(file_names))


user_dir = R'D:\SteamLibrary\steamapps\workshop\content\431960'
# copy_file(user_dir, 'D:/999')
# copy_file_max(user_dir, 'D:/999')
copy_file_all(user_dir, 'D:/999')
