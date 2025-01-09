import argparse
import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Generate label stat info")
parser.add_argument("-d",
                    "--datadir",
                    default="",
                    help="path to load data",
                    type=str,
                    )
parser.add_argument("-n",
                    "--nprocs",
                    default=16,
                    help="Number of processes",
                    type=int,
                    )
parser.add_argument("-o",
                    "--output_dir",
                    default="",
                    help="path to save label info",
                    type=str,
                    )
args = parser.parse_args()
imgdir = os.path.join(args.datadir, 'RAND_CITYSCAPES', 'RGB')
labdir = os.path.join(args.datadir, 'RAND_CITYSCAPES', 'GT', 'LABELS')
labfiles = os.listdir(labdir)
nprocs = args.nprocs
savedir = args.output_dir

id_to_trainid = {
    3: 0,    # road
    4: 1,    # sidewalk
    2: 2,    # building
    21: 3,   # wall
    5: 4,    # fence
    7: 5,    # pole
    15: 6,   # traffic light
    9: 7,    # traffic sign
    6: 8,    # vegetation
    1: 9,    # sky
    10: 10,  # person
    17: 11,  # rider
    8: 12,   # car
    19: 13,  # bus
    12: 14,  # motorcycle
    11: 15,  # bicycle
}

def _foo(i):
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = dict()
    labfile = labfiles[i]
    file_to_label[labfile] = []
    
    try:
        # Read label using cv2 and take last channel
        label_data = cv2.imread(os.path.join(labdir, labfile), cv2.IMREAD_UNCHANGED)[:, :, -1]
        label = np.unique(label_data)
        
        found_labels = []
        for lab in label:
            if lab in id_to_trainid:
                l = id_to_trainid[lab]
                label_to_file[l].append(labfile)
                file_to_label[labfile].append(l)
                found_labels.append(lab)
                
        if not found_labels:
            print(f"Warning: No valid labels found in {labfile}. Unique values: {label}")
            
    except Exception as e:
        print(f"Error processing file {labfile}: {str(e)}")
        return [[] for _ in range(len(id_to_trainid.keys()))], {}
        
    return label_to_file, file_to_label

def main():
    print(f"Processing labels from directory: {labdir}")
    print(f"Number of label files to process: {len(labfiles)}")
    print(f"Using {nprocs} processes")
    print(f"Looking for {len(id_to_trainid)} classes")
    
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e: [] for e in os.listdir(imgdir)}

    if nprocs == 1:
        print("Running in single process mode")
        for i in tqdm(range(len(labfiles))):
            l2f, f2l = _foo(i)
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l:
                file_to_label[fname].extend(f2l[fname])
    else:
        print("Running in parallel mode")
        with Pool(nprocs) as p:
            r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
            
        for l2f, f2l in r:
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l:
                file_to_label[fname].extend(f2l[fname])

    print("\nLabel statistics:")
    for lab in range(len(label_to_file)):
        print(f"Class {lab} (train_id): {len(label_to_file[lab])} images")
    
    print("\nSaving results...")
    with open(os.path.join(savedir, 'synthia_label_info.p'), 'wb') as f:
        pickle.dump((label_to_file, file_to_label), f)
    
    print("Done!")
    
    print("\nVerifying saved file...")
    with open(os.path.join(savedir, 'synthia_label_info.p'), 'rb') as f:
        l2f, f2l = pickle.load(f)
        print("Verification successful!")
        print("\nFinal statistics after loading:")
        for lab in range(len(l2f)):
            print(f"Class {lab} (train_id): {len(l2f[lab])} images")

if __name__ == "__main__":
    main()
