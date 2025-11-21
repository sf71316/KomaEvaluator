import os
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil
import hashlib

def check_image(file_path):
    """
    檢查單張圖片是否損壞，並返回結果。
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # 驗證圖片完整性
            
            # 額外檢查：嘗試重新開啟並讀取數據，因為 verify 只有檢查 header
            with Image.open(file_path) as img2:
                img2.load() # 強制解碼圖片數據
                
                # 檢查是否為極端長寬比 (例如條漫)
                w, h = img2.size
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > 5: # 長寬比超過 1:5
                    return 'ratio', file_path
                    
                # 檢查是否過小
                if w < 64 or h < 64:
                    return 'small', file_path
                    
        return 'ok', file_path
    except Exception as e:
        return 'corrupt', file_path

def calculate_hash(file_path):
    """計算檔案的 MD5 Hash"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest(), file_path
    except Exception:
        return None, file_path

def main():
    parser = argparse.ArgumentParser(description='資料集健康檢查工具 (多執行緒版)')
    parser.add_argument('--data_dir', type=str, required=True, help='要檢查的資料集根目錄')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='執行緒數量')
    parser.add_argument('--move_corrupt', type=str, default=None, help='將損壞檔案移動到的目錄 (可選)')
    parser.add_argument('--delete_corrupt', action='store_true', help='直接刪除損壞檔案 (慎用)')
    parser.add_argument('--check_duplicates', action='store_true', help='檢查重複檔案 (基於 MD5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"錯誤: 目錄 {args.data_dir} 不存在")
        return

    # 1. 收集所有圖片路徑
    print(f"正在掃描目錄: {args.data_dir} ...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"共找到 {len(image_files)} 張圖片。")
    
    # 2. 多執行緒檢查完整性
    print(f"開始完整性檢查 (使用 {args.num_workers} 執行緒)...")
    corrupt_files = []
    small_files = []
    ratio_files = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(check_image, f) for f in image_files]
        for future in tqdm(as_completed(futures), total=len(image_files), unit='img'):
            status, path = future.result()
            if status == 'corrupt':
                corrupt_files.append(path)
            elif status == 'small':
                small_files.append(path)
            elif status == 'ratio':
                ratio_files.append(path)

    # 3. 報告結果
    print("\n" + "="*40)
    print("檢查報告")
    print("="*40)
    print(f"總檢查數量: {len(image_files)}")
    print(f"損壞檔案: {len(corrupt_files)}")
    print(f"過小檔案 (<64px): {len(small_files)}")
    print(f"極端長寬比 (>1:5): {len(ratio_files)}")
    
    if corrupt_files:
        print("\n[損壞檔案列表]:")
        for f in corrupt_files[:10]: # 只顯示前10個
            print(f"  - {f}")
        if len(corrupt_files) > 10:
            print(f"  ... 以及其他 {len(corrupt_files)-10} 個")

    # 4. 處理損壞檔案
    if corrupt_files:
        if args.move_corrupt:
            os.makedirs(args.move_corrupt, exist_ok=True)
            print(f"\n正在移動損壞檔案至: {args.move_corrupt}")
            for f in corrupt_files:
                try:
                    filename = os.path.basename(f)
                    # 保持原始子目錄結構有點複雜，這裡簡單移動到根目錄，避免檔名衝突加上 hash
                    shutil.move(f, os.path.join(args.move_corrupt, f"{hashlib.md5(f.encode()).hexdigest()[:6]}_{filename}"))
                except Exception as e:
                    print(f"移動失敗 {f}: {e}")
        elif args.delete_corrupt:
            print("\n正在刪除損壞檔案...")
            for f in corrupt_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"刪除失敗 {f}: {e}")
            print("刪除完成。")
        else:
            print("\n提示: 使用 --move_corrupt <dir> 或 --delete_corrupt 來自動處理這些檔案。")

    # 5. 重複檔案檢查 (如果啟用)
    if args.check_duplicates:
        print("\n正在檢查重複檔案 (MD5)...")
        hashes = {}
        duplicates = []
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(calculate_hash, f) for f in image_files]
            for future in tqdm(as_completed(futures), total=len(image_files), unit='img'):
                file_hash, path = future.result()
                if file_hash:
                    if file_hash in hashes:
                        duplicates.append((path, hashes[file_hash]))
                    else:
                        hashes[file_hash] = path
        
        print(f"發現 {len(duplicates)} 組重複圖片。")
        if duplicates:
             print("部分重複範例:")
             for dup in duplicates[:5]:
                 print(f"  - {dup[0]} == {dup[1]}")

if __name__ == '__main__':
    main()
