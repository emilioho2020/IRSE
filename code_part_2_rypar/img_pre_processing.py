import numpy as np
import pandas as pd



# GENERATE TRAIN, VAL, TEST SET SPLITS OF GIVEN CSV FILE WITH IMAGE FEATURES AND SAVE IT AS NPY FILE
def generate_npy(filepath, targetFolder):
    splits = ['train', 'test', 'val']
    data = pd.read_csv(filepath, sep=' ', header=None)
    data[0] = data[0].map(lambda x: x.split('.')[0])  # remove .jpg ending
    
    for split in splits:
        split_idx = [im.strip() for im in open(ARGS.feat_path + split + '.txt', 'r')]
        data_split = data.loc[data[0].isin(split_idx)].copy()
        sorterIndex = dict(zip(split_idx, range(len(split_idx))))  # sort by given split
        # Generate a rank column that will be used to sort the dataframe numerically
        data_split['rank'] = data_split[0].map(sorterIndex)
        data_split.sort_values(['rank'], ascending = [True], inplace = True)
        data_split.drop(['rank', 0], 1, inplace = True)  # drop rank and name of image columns
        data_split = data_split.to_numpy(dtype=np.float32)
        np.save(targetFolder + split + '_im_features.npy', data_split)
        print(split, 'saved with shape', data_split.shape)