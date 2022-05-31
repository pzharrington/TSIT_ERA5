""" 
    simple script to copy weatherbench data
    and split it into train, validation, and test
    according to the year
"""
import os, sys, shutil

def create_dirs(nm):
    if not os.path.exists(nm):
        os.makedirs(nm)

this_dir = os.getcwd() # where do you want to store the data?
train_dir = os.path.join(this_dir, "train")
val_dir = os.path.join(this_dir, "validation")
test_dir = os.path.join(this_dir, "test")

create_dirs(train_dir)
create_dirs(val_dir)
create_dirs(test_dir)

data_dir = "/global/cfs/cdirs/dasrepo/shashank/weatherbench/"
variables = ["geopotential_500", "temperature_850"]

valid_time = ["2016"]
test_time = ["2017", "2018"]

for v in variables:
    var_dir = os.path.join(data_dir, v)
    if not os.path.exists(var_dir):
        print("no data for variable {}".format(v))
        continue

    val_dir_v = os.path.join(val_dir, v)
    train_dir_v = os.path.join(train_dir, v)
    test_dir_v = os.path.join(test_dir, v)
    create_dirs(val_dir_v)
    create_dirs(train_dir_v)
    create_dirs(test_dir_v)
   
    files = os.listdir(var_dir)
    for f in files:
        if f.find(".zip") is not -1:
            continue

        is_valid = False
        is_test = False
        for vt in valid_time:
            if f.find(vt) is not -1:
                print("copying {} to validation dir".format(f))
                src = os.path.join(var_dir, f)
                dst = os.path.join(val_dir_v, f)
                shutil.copy(src, dst)
                is_valid = True
                break

        if is_valid:
            continue

        for tt in test_time:
            if f.find(tt) is not -1:
                print("copying {} to testing dir".format(f))
                src = os.path.join(var_dir, f)
                dst = os.path.join(test_dir_v, f)
                shutil.copy(src, dst)
                is_test = True
                break
        
        if is_test:
            continue
           
        print("copying {} to training dir".format(f))
        src = os.path.join(var_dir, f)
        dst = os.path.join(train_dir_v, f)
        shutil.copy(src, dst)
