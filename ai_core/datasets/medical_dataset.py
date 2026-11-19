import cv2
import pandas as pd
from torch.utils.data import Dataset
class MedicalLesionDataset(Dataset):
    def __init__(self, csv_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform 
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        mask_path = self.df.iloc[idx]["mask_path"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype("float32") # Đưa về nhãn nhị phân 0 và 1
        if self.transform:
            augmented = self.transform(image=image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        return image, mask