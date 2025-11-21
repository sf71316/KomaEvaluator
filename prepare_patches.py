import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_high_quality_patch(patch, std_threshold=30, edge_threshold=0.05):
    """
    判斷切塊是否為高品質 (非純色、非簡單線條)。
    """
    try:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < std_threshold:
            return False
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        
        if edge_density < edge_threshold:
            return False
            
        return True
    except Exception:
        return False

def extract_patches_from_image(img_path, patch_size=224, attempts=10):
    """從單張圖片嘗試提取一個高品質切塊"""
    try:
        # 支援中文路徑讀取
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            return None

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        h, w = img.shape[:2]
        
        if h < patch_size or w < patch_size:
            return None
            
        for _ in range(attempts):
            y = random.randint(0, h - patch_size)
            x = random.randint(0, w - patch_size)
            
            patch = img[y:y+patch_size, x:x+patch_size]
            
            if is_high_quality_patch(patch):
                return patch
                
    except Exception as e:
        # print(f"Error reading {img_path}: {e}")
        pass
        
    return None

def process_artist_task(args):
    """
    處理單一位畫師的任務 (用於 ProcessPoolExecutor)
    """
    artist_name, image_files, dst_dir, target_count, patch_size = args
    
    artist_dir = os.path.join(dst_dir, artist_name)
    os.makedirs(artist_dir, exist_ok=True)
    
    saved_count = 0
    max_loops = 5  # 避免圖片太少無限循環，最多遍歷所有圖片 5 次
    current_loop = 0
    
    # 如果已經有檔案，先計算 (支援斷點續傳)
    existing_files = [f for f in os.listdir(artist_dir) if f.endswith('.jpg')]
    saved_count = len(existing_files)
    
    if saved_count >= target_count:
        return f"{artist_name}: 已達標 ({saved_count})"

    while saved_count < target_count and current_loop < max_loops:
        current_loop += 1
        random.shuffle(image_files) # 每輪隨機打亂
        
        progress_made = False
        
        for file_path in image_files:
            if saved_count >= target_count:
                break
                
            patch = extract_patches_from_image(file_path, patch_size)
            
            if patch is not None:
                # 使用 UUID 或 hash 避免多進程或多次執行時檔名衝突，這裡簡單用流水號
                # 注意：多進程同時寫同一個資料夾可能會有 race condition，但這裡是每個進程負責一個資料夾，所以安全
                save_name = f"patch_{saved_count:05d}.jpg"
                save_path = os.path.join(artist_dir, save_name)
                
                is_success, im_buf = cv2.imencode(".jpg", patch)
                if is_success:
                    im_buf.tofile(save_path)
                    saved_count += 1
                    progress_made = True
        
        if not progress_made:
            # 如果這一整輪都沒有提取到任何新 patch，可能是圖片品質太差，提前結束
            break
            
    return f"{artist_name}: 完成，共產生 {saved_count} 張 (目標 {target_count})"

def main():
    parser = argparse.ArgumentParser(description='多進程漫畫質感切塊提取工具')
    parser.add_argument('--src_dir', type=str, required=True, help='原始漫畫圖檔根目錄 (Manga_Dataset_Clean)')
    parser.add_argument('--dst_dir', type=str, required=True, help='輸出目錄 (Intermediate_Patches)')
    parser.add_argument('--target_count', type=int, default=400, help='每位畫師的目標切塊數量')
    parser.add_argument('--patch_size', type=int, default=224, help='切塊大小')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='並行處理的畫師數量')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png,webp,bmp', help='支援的副檔名')
    
    args = parser.parse_args()
    
    valid_exts = tuple(f".{ext.strip()}" for ext in args.extensions.split(','))
    
    if not os.path.exists(args.src_dir):
        print(f"Error: Source directory '{args.src_dir}' not found.")
        return

    # 1. 收集所有畫師及其圖片
    print("正在掃描資料集結構...")
    artist_tasks = []
    
    # 找出所有可能的畫師名稱 (聯集 train/val/test)
    all_artists = set()
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(args.src_dir, split)
        if os.path.exists(split_path):
            for name in os.listdir(split_path):
                if os.path.isdir(os.path.join(split_path, name)):
                    all_artists.add(name)
    
    sorted_artists = sorted(list(all_artists))
    print(f"共發現 {len(sorted_artists)} 位畫師。")

    # 準備任務參數
    for artist in sorted_artists:
        image_files = []
        # 跨分割區收集該畫師的所有圖片
        for split in ['train', 'val', 'test']:
            artist_path = os.path.join(args.src_dir, split, artist)
            if os.path.exists(artist_path):
                for root, dirs, files in os.walk(artist_path):
                    for file in files:
                        if file.lower().endswith(valid_exts):
                            image_files.append(os.path.join(root, file))
        
        if image_files:
            # 打包參數: (artist_name, files, dst_dir, target, size)
            artist_tasks.append((artist, image_files, args.dst_dir, args.target_count, args.patch_size))
        else:
            print(f"警告: 畫師 {artist} 沒有找到任何圖片，跳過。")

    # 2. 多進程執行
    print(f"開始提取切塊 (使用 {args.num_workers} 執行緒)...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_artist_task, task) for task in artist_tasks]
        
        # 使用 tqdm 顯示進度條 (以畫師為單位)
        for future in tqdm(as_completed(futures), total=len(artist_tasks), unit='artist'):
            result = future.result()
            # print(result) # 可選：顯示詳細日誌

    print("\n所有畫師處理完成！")

if __name__ == '__main__':
    main()