import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_IMG_DIR = "ai_core/data/raw/images"
RAW_MASK_DIR = "ai_core/data/raw/masks"
OUT_DIR = "ai_core/data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

image_files = sorted(glob.glob(os.path.join(RAW_IMG_DIR, "*.jpg")))
pairs = []

for img_path in image_files:
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # VD: IDRiD_01
    matched_masks = glob.glob(os.path.join(RAW_MASK_DIR, f"{base_name}_*.tif"))
    if matched_masks:
        pairs.append({
            "image_path": img_path.replace("\\", "/"),
            "mask_path": matched_masks[0].replace("\\", "/")
        })

df = pd.DataFrame(pairs)
print(f"Tổng số cặp ảnh - mask khớp thành công: {len(df)}")

# Chia 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print(f"Đã lưu: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")