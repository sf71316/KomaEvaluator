import cv2
import os
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def process_single_image(args):
    """
    處理單張圖片：讀取 -> 偵測 -> 裁切 -> 儲存
    """
    src_path, dst_subdir, cascade_file, min_size, scale_factor, min_neighbors, padding = args
    
    # 每個進程需要重新加載 cascade，因為它不能被 pickle
    cascade = cv2.CascadeClassifier(cascade_file)
    saved_count = 0
    
    try:
        # 使用 numpy 讀取以支援中文路徑
        img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), -1)
        if img is None: return 0
        
        # 處理通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size)
        )
        
        if len(faces) == 0: return 0

        filename_no_ext = os.path.splitext(os.path.basename(src_path))[0]
        
        for i, (x, y, w, h) in enumerate(faces):
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img.shape[1], x + w + pad_w)
            y2 = min(img.shape[0], y + h + pad_h)
            
            face_img = img[y1:y2, x1:x2]
            
            # 確保輸出目錄存在
            os.makedirs(dst_subdir, exist_ok=True)
            
            save_name = f"{filename_no_ext}_face_{i}.jpg"
            save_path = os.path.join(dst_subdir, save_name)
            
            is_success, im_buf = cv2.imencode(".jpg", face_img)
            if is_success:
                im_buf.tofile(save_path)
                saved_count += 1
                
        return saved_count
        
    except Exception as e:
        return 0

def main():
    parser = argparse.ArgumentParser(description='多進程動漫人臉裁切工具 (v2)')
    parser.add_argument('--src_dir', required=True, help='來源圖片目錄')
    parser.add_argument('--dst_dir', required=True, help='輸出目錄')
    parser.add_argument('--cascade', default='lbpcascade_animeface.xml')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='進程數量')
    
    # 裁切參數
    parser.add_argument('--padding', type=float, default=0.2)
    parser.add_argument('--min_size', type=int, default=40)
    parser.add_argument('--scale_factor', type=float, default=1.05)
    parser.add_argument('--min_neighbors', type=int, default=7)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cascade):
        print(f"錯誤: 找不到 Cascade 檔案: {args.cascade}")
        return

    # 1. 收集所有任務
    print(f"正在掃描目錄: {args.src_dir} ...")
    tasks = []
    
    for root, dirs, files in os.walk(args.src_dir):
        # 簡單判斷是否在 artist 資料夾層級
        if root == args.src_dir: 
            continue
            
        # 計算目標子目錄
        rel_path = os.path.relpath(root, args.src_dir)
        dst_subdir = os.path.join(args.dst_dir, rel_path)
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                src_path = os.path.join(root, file)
                # 打包參數
                tasks.append((src_path, dst_subdir, args.cascade, args.min_size, args.scale_factor, args.min_neighbors, args.padding))

    print(f"共發現 {len(tasks)} 張圖片，準備使用 {args.num_workers} 個進程進行處理。")
    
    # 2. 多進程執行
    total_faces = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), unit='img'):
            total_faces += future.result()
            
    print(f"\n裁切完成！共提取 {total_faces} 張人臉。")
    print("注意：此階段不進行數量平衡，將保留所有裁切到的人臉。")

if __name__ == '__main__':
    main()