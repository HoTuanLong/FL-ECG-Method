import os, sys
from libs import *

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_dir, 
    ):
        self.df_path, self.data_dir,  = df_path, data_dir, 
        self.df = pd.read_csv(self.df_path)

        self.tgt_lens = self.df.iloc[:, -30:].sum().values.tolist()

    def __len__(self, 
    ):
        return len(self.df)

    def __getitem__(self, 
        index, 
    ):
        row = self.df.iloc[index]
        tgt = row.values[-30:].astype("float64")

        ecg = sio.loadmat("{}/{}.mat".format(self.data_dir, row["Id"]))["val"]
        ecg = sequence.pad_sequences(ecg, 5000, "float64", 
            "post", "post", 
        )

        return ecg, tgt