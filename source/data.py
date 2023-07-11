import os, sys
from libs import *

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_path, 
    ):
        self.df_path, self.data_path,  = df_path, data_path, 
        self.df = pd.read_csv(self.df_path)

    def __len__(self, 
    ):
        return len(self.df)

    def __getitem__(self, 
        index, 
    ):
        row = self.df.iloc[index]

        ecg = sio.loadmat("{}/{}.mat".format(self.data_path, row["Id"]))["val"]
        ecg = sequence.pad_sequences(ecg, 5000, "float64", 
            "post", "post", 
        )
        ecg, tgt = torch.tensor(ecg).float(), torch.tensor(row.values[-30:].astype("float64")).float()

        return ecg, tgt