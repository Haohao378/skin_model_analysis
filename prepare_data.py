import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  

SOURCE_DIR = "data/HAM10000_original"
METADATA_PATH = os.path.join(SOURCE_DIR, "HAM10000_metadata.csv")

OUTPUT_DIR = "data/formatted"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def main():

    print(f"正在读取元数据: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
   
    print("正在扫描所有图片路径...")
    image_path_dict = {}

    parts = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    
    for part in parts:
        part_folder = os.path.join(SOURCE_DIR, part)
        if not os.path.exists(part_folder):
            print(f"⚠️ 警告: 文件夹 {part_folder} 不存在，跳过。")
            continue
            
        for img_name in os.listdir(part_folder):
            if img_name.endswith('.jpg'):
                img_id = os.path.splitext(img_name)[0]
                image_path_dict[img_id] = os.path.join(part_folder, img_name)

    print(f"共找到 {len(image_path_dict)} 张图片。")

    train_df, val_df = train_test_split(
        df, 
        train_size=TRAIN_RATIO, 
        stratify=df['dx'], 
        random_state=RANDOM_SEED
    )

    print(f"划分完成: 训练集 {len(train_df)} 张, 验证集 {len(val_df)} 张")

    def copy_files(dataframe, split_type):
        print(f"\n正在处理 {split_type} 数据集...")
        
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            img_id = row['image_id']
            label = row['dx'] 

            if img_id not in image_path_dict:
                print(f"❌ 错误: 找不到图片 {img_id}")
                continue
            src_path = image_path_dict[img_id]

            dest_folder = os.path.join(OUTPUT_DIR, split_type, label)
            os.makedirs(dest_folder, exist_ok=True)

            dest_path = os.path.join(dest_folder, f"{img_id}.jpg")

            shutil.copy2(src_path, dest_path)

    copy_files(train_df, "train")
    copy_files(val_df, "val")

    print("\n✅ 数据处理完成！")
    print(f"数据已保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    try:
        import sklearn
    except ImportError:
        print("请先安装 scikit-learn: pip install scikit-learn pandas tqdm")
        exit(1)
        
    main()