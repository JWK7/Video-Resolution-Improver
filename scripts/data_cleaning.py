import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from utility.online_load import load_gif

def clean_file(**kwargs):
    data = pd.read_csv(kwargs["data_directory"], header = None)[0]
    if os.path.exists(kwargs["cleaned_filename"]):
        cleaned_data = pd.read_csv(kwargs["cleaned_filename"])
        starting_index = len(cleaned_data)
        print(f"Continueing from {starting_index}...")
    else:
        cleaned_data = pd.DataFrame(columns=["url","frames","height","width"])
        print("Starting fresh...")
        starting_index = 0

    data = data[starting_index:]


    batch_data = pd.DataFrame(index=range(kwargs['batch_size']),columns=["url","frames","height","width"])
    i = 0
    for url in data:
        try:
            frames,height,width = load_gif(url).shape
            batch_data.loc[i] = [url,frames,height,width]
        except:
            pass
        
        i+=1
        
        if kwargs['batch_size']-i == 0:
            cleaned_data = pd.concat([cleaned_data,batch_data],ignore_index=True)
            cleaned_data.to_csv(kwargs["cleaned_filename"],index=False)
            batch_data = pd.DataFrame(index=range(kwargs['batch_size']),columns=["url","frames","height","width"])
            i = 0

def main():
    data_directory = "data/train.txt"
    cleaned_filename = "data/cleaned_train.csv"
    clean_file(data_directory = data_directory,cleaned_filename = cleaned_filename,batch_size = 10)

if __name__ == "__main__":
    main()