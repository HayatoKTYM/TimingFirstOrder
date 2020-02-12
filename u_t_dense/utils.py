import numpy as np
import pandas as pd
import glob

def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    """
    if type(np_list) == type('hello'): # str 
        return pd.DataFrame(np.load(np_list))
    else: #np.load 済みなら
        return pd.DataFrame(np_list)
        
def setup():
    img_middle_feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2020/img_middle64/*npy'))
    spec_middle_feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2020/turn_middle64/*npy'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2020/feature/*csv'))
    print(f'file length is {len(img_middle_feature_files)} and {len(spec_middle_feature_files)} and {len(feature_files)}')
    df_list = []

    for i in range(len(feature_files)):
        df = pd.read_csv(feature_files[i])
        img = np_to_dataframe(img_middle_feature_files[i])
        turn = np_to_dataframe(spec_middle_feature_files[i])
        
        df = pd.concat([df,img,turn],axis=1)
        df_list.append(df)

    return df_list