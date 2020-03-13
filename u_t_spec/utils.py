import numpy as np
import pandas as pd
import glob

def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    """
    if type(np_list) == type('hello'): # str 
        np_list = np.load(np_list)
        np_list = np_list[:len(np_list)//2*2] #奇数個なら偶数個に
        np_list = np_list.reshape(-1,256) #20fps > 10fps
        return pd.DataFrame(np_list)
    else: #np.load 済みなら
        return pd.DataFrame(np_list)
        
def setup(dense_flag=False):
    img_middle_feature_files = sorted(glob.glob('/mnt/aoni04/katayama/spec/*npy'))
    feature_files = sorted(glob.glob('/mnt/aoni04/katayama/vad/*csv'))
    print(f'file length is {len(img_middle_feature_files)} and {len(feature_files)}')
    df_list = []

    for i in range(0,len(feature_files)):
        df = pd.read_csv(feature_files[i])
        img = np_to_dataframe(img_middle_feature_files[2*i])
        imgB = np_to_dataframe(img_middle_feature_files[2*i+1])
        
        df = df[:min([len(df),len(img)//2])] # vad_file と長さ調整
        img = img[:min([len(df),len(img)])]
        imgB = imgB[:min([len(df),len(imgB)])]

        df = pd.concat([df,img,imgB],axis=1)
        if not dense_flag:
            reset_array = [-1] * len(df.columns)
            df.loc['reset'] = reset_array 
            df_list.append(df)
        else:
            df_list.append(df)

    return df_list
