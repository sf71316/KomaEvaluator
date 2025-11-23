import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import psutil # 導入 psutil

def set_low_priority():
    try:
        p = psutil.Process(os.getpid())
        if os.name == 'nt': # Windows
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else: # Linux
            p.nice(10) # 0 is normal, 19 is lowest priority
        print(f"[優先度] 已將進程優先度調降為: BELOW_NORMAL")
    except Exception as e:
        print(f"[優先度] 無法調整優先度: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', nargs='+', required=True, help='要合併的來源目錄列表 (e.g. Intermediate_Faces Intermediate_Patches)')
    parser.add_argument('--dst_dir', required=True, help='最終輸出的資料集目錄 (e.g. Manga_Dataset_Mixed)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='驗證集比例 (預設 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='測試集比例 (預設 0.15)')
    parser.add_argument('--low_priority', action='store_true', help='調降進程優先度 (背景運行)') # 新增參數
    args = parser.parse_args()

    if args.low_priority:
        set_low_priority()

    # 建立目標結構 - 包含 train, val, test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.dst_dir, split), exist_ok=True)

    # 掃描所有來源目錄，建立畫師名單
    artists = set()
    for d in args.dirs:
        if not os.path.exists(d): continue
        
        # 檢查是否為分割結構 (含有 train/val/test 子目錄)
        subdirs = [sd for sd in os.listdir(d) if os.path.isdir(os.path.join(d, sd))]
        is_split_structure = any(s in subdirs for s in ['train', 'val', 'test'])
        
        if is_split_structure:
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(d, split)
                if os.path.exists(split_path):
                    for artist in os.listdir(split_path):
                        if os.path.isdir(os.path.join(split_path, artist)):
                            artists.add(artist)
        else:
            for artist in subdirs:
                artists.add(artist)

    print(f"偵測到 {len(artists)} 位畫師，開始合併與分割 (Train/Val/Test)...")

    # 第一階段：收集所有圖片並統計數量
    artist_images = {}
    min_count = float('inf')

    for artist in artists:
        all_images = []
        
        # 從所有來源目錄收集該畫師的圖片
        for d in args.dirs:
            if not os.path.exists(d): continue
            
            # 檢查是否為分割結構
            subdirs = [sd for sd in os.listdir(d) if os.path.isdir(os.path.join(d, sd))]
            is_split_structure = any(s in subdirs for s in ['train', 'val', 'test'])
            
            search_paths = []
            if is_split_structure:
                for split in ['train', 'val', 'test']:
                    search_paths.append(os.path.join(d, split, artist))
            else:
                search_paths.append(os.path.join(d, artist))
                
            for src_artist_path in search_paths:
                if os.path.exists(src_artist_path):
                    for f in os.listdir(src_artist_path):
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                            all_images.append(os.path.join(src_artist_path, f))
        
        # 隨機打亂
        random.shuffle(all_images)
        artist_images[artist] = all_images
        
        if len(all_images) < min_count:
            min_count = len(all_images)

    print(f"最少圖片數量: {min_count}。將執行平衡邏輯 (容忍度 20%)...")
    max_allowed = int(min_count * 1.2)

    # 第二階段：執行平衡與分割
    for artist, all_images in tqdm(artist_images.items(), desc="處理畫師", unit="位"):
        original_count = len(all_images)
        
        if len(all_images) > max_allowed:
            all_images = all_images[:max_allowed]
            print(f"  - {artist}: 削減 {original_count} -> {len(all_images)} (達到平衡上限)")
        
        # 分割
        num_total = len(all_images)
        num_val = int(num_total * args.val_ratio)
        num_test = int(num_total * args.test_ratio)
        num_train = num_total - num_val - num_test # num_train 是剩下的
        
        train_files = all_images[:num_train]
        val_files = all_images[num_train : num_train + num_val]
        test_files = all_images[num_train + num_val:]
        
        # 複製檔案
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dst_artist_path = os.path.join(args.dst_dir, split, artist)
            os.makedirs(dst_artist_path, exist_ok=True)
            
            for src_path in files:
                # 為了避免檔名衝突 (例如兩個來源都有 001.jpg)，加上來源前綴
                src_folder_name = Path(src_path).parent.parent.name # e.g. Intermediate_Faces
                file_name = Path(src_path).name
                new_name = f"{src_folder_name}_{file_name}"
                shutil.copy(src_path, os.path.join(dst_artist_path, new_name))
                
        # print(f"  - {artist}: Train {len(train_files)} / Val {len(val_files)} / Test {len(test_files)} (Total: {len(all_images)})")

if __name__ == '__main__':
    main()