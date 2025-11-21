import os
import shutil
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageStat
from tqdm import tqdm
import random

def calculate_complexity(img):
    """
    計算圖片的複雜度 (線條密度)。
    使用邊緣檢測濾鏡，計算邊緣像素的平均強度。
    """
    # 轉為灰階
    gray = img.convert('L')
    # 應用邊緣檢測
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # 計算邊緣的平均強度 (0-255)
    stat = ImageStat.Stat(edges)
    avg_edge_intensity = stat.mean[0]
    
    #正規化到 0-1 之間 (假設最大強度不會全滿，但除以 255 是標準做法)
    density = avg_edge_intensity / 255.0
    return density

def augment_dataset(src_dir, dst_dir, crop_size=224, stride=112, threshold=0.05, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    """
    增強資料集：保留原圖並進行滑動視窗裁切。
    """
    if os.path.exists(dst_dir):
        print(f"警告: 目標目錄 {dst_dir} 已存在。")
        response = input("是否刪除並重建？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(dst_dir)
        else:
            print("已取消。")
            return

    print(f"開始處理資料集...")
    print(f"來源: {src_dir}")
    print(f"目標: {dst_dir}")
    print(f"裁切大小: {crop_size}x{crop_size}, 步長: {stride}")
    print(f"複雜度過濾閾值: {threshold}")

    processed_counts = {}

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(extensions):
                src_path = os.path.join(root, file)
                
                # 構建目標路徑
                rel_path = os.path.relpath(root, src_dir)
                dst_subdir = os.path.join(dst_dir, rel_path)
                os.makedirs(dst_subdir, exist_ok=True)
                
                artist_name = os.path.basename(dst_subdir)
                if artist_name not in processed_counts:
                    processed_counts[artist_name] = {'original': 0, 'crops': 0, 'skipped': 0}

                try:
                    with Image.open(src_path) as img:
                        img = img.convert('RGB')
                        filename_no_ext = os.path.splitext(file)[0]

                        # 1. 保存原圖 (縮放至稍大於 crop_size 以保留整體感，或保持原樣)
                        # 這裡我們保持原樣，訓練時 transforms 會處理縮放
                        # 為了統一，我們將原圖複製過去
                        shutil.copy2(src_path, os.path.join(dst_subdir, file))
                        processed_counts[artist_name]['original'] += 1

                        # 2. 進行裁切 (僅針對訓練集，通常 val/test 不建議裁切以免汙染評估)
                        # 這裡簡單判斷：如果路徑包含 'train' 才裁切，或者全部裁切由使用者決定
                        # 為了通用性，我們對所有輸入都裁切，但建議使用者只對 train 資料夾執行此腳本
                        
                        w, h = img.size
                        
                        # 如果圖片比裁切框還小，就跳過裁切
                        if w < crop_size or h < crop_size:
                            continue

                        # 滑動視窗
                        count_crops = 0
                        for y in range(0, h - crop_size + 1, stride):
                            for x in range(0, w - crop_size + 1, stride):
                                crop = img.crop((x, y, x + crop_size, y + crop_size))
                                
                                # 檢查複雜度
                                complexity = calculate_complexity(crop)
                                
                                if complexity >= threshold:
                                    save_name = f"{filename_no_ext}_crop_{x}_{y}.jpg"
                                    crop.save(os.path.join(dst_subdir, save_name), quality=90)
                                    count_crops += 1
                                else:
                                    processed_counts[artist_name]['skipped'] += 1
                        
                        processed_counts[artist_name]['crops'] += count_crops

                except Exception as e:
                    print(f"處理 {src_path} 時發生錯誤: {e}")

    print("\n處理完成！統計結果：")
    total_images = 0
    print(f"{'Artist':<30} | {'Original':<10} | {'New Crops':<10} | {'Skipped (Low Info)':<20} | {'Total':<10}")
    print("-" * 90)
    for artist, counts in processed_counts.items():
        total = counts['original'] + counts['crops']
        total_images += total
        print(f"{artist:<30} | {counts['original']:<10} | {counts['crops']:<10} | {counts['skipped']:<20} | {total:<10}")
    print("-" * 90)
    print(f"總圖片數量: {total_images}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='資料集增強工具：自動裁切與過濾')
    parser.add_argument('--src_dir', type=str, required=True, help='原始資料集路徑 (例如: Manga_Dataset_7_Artists/train)')
    parser.add_argument('--dst_dir', type=str, required=True, help='輸出資料集路徑 (例如: Manga_Dataset_Augmented/train)')
    parser.add_argument('--threshold', type=float, default=0.035, help='複雜度過濾閾值 (0.0-1.0)，越高品質越高。建議 0.03~0.05')
    parser.add_argument('--stride', type=int, default=150, help='滑動視窗的步長 (越小重疊越多，產生的圖越多)')
    
    args = parser.parse_args()
    
    augment_dataset(args.src_dir, args.dst_dir, threshold=args.threshold, stride=args.stride)
