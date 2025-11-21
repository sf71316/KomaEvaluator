import os
import zipfile
import shutil
import random
from collections import defaultdict
import argparse
from PIL import Image
import psutil
import time

def is_image_valid(image_path, min_aspect_ratio, max_aspect_ratio):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if height == 0: return False
            aspect_ratio = width / height
            return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
    except Exception as e:
        # print(f"警告：無法讀取圖片 {image_path}，已跳過。錯誤：{e}")
        return False

def load_trained_history(history_file='trained_history.txt'):
    """讀取已訓練過的作者清單"""
    if not os.path.exists(history_file):
        return set()
    
    history_set = set()
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    history_set.add(line)
    except Exception as e:
        print(f"讀取歷史紀錄失敗: {e}")
        
    return history_set

def get_user_confirmed_allowlist(src_dir, allowlist_file='whitelist.txt'):
    """
    互動式取得白名單：
    1. 讀取舊名單
    2. 掃描新作者並附加
    3. 提示編輯
    4. 讀取最終名單
    """
    if not os.path.exists(src_dir):
        print(f"錯誤: 來源目錄 {src_dir} 不存在。")
        return []

    # 1. 掃描現有目錄
    all_artists_in_dir = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    
    if not all_artists_in_dir:
        print("錯誤: 來源目錄下沒有發現任何資料夾。")
        return []

    new_artists = []
    
    # 2. 檢查並更新白名單檔案
    if not os.path.exists(allowlist_file):
        print(f"\n[初始化] 未發現白名單檔案，正在建立 {allowlist_file} ...")
        new_artists = all_artists_in_dir # 全都是新的
        try:
            with open(allowlist_file, 'w', encoding='utf-8') as f:
                f.write("# ==========================================\n")
                f.write("# 訓練白名單設定檔\n")
                f.write("# 請保留您想要處理的作者資料夾名稱\n")
                f.write("# 若要略過某位作者，請在行首加上井字號 (#)\n") # 修正說明文字
                f.write("# ==========================================\n\n")
                for artist in new_artists:
                    f.write(f"{artist}\n")
            
            print(f"已成功建立預設白名單: {os.path.abspath(allowlist_file)}")
            
        except IOError as e:
            print(f"無法寫入白名單檔案: {e}")
            return []
            
        # 第一次建立，必定需要確認
        print("="*60)
        print(f"我幫你建了一份白名單 ({allowlist_file})，你去看一下哪些要訓練，哪些是不要訓練。")
        print("="*60)
        input("編輯完成並存檔後，請按 Enter 鍵繼續程式...")
        
    else:
        # 檔案存在，檢查是否有新作者
        try:
            existing_lines = []
            existing_artists_in_file = set() # 記錄檔案中所有作者 (包含註解掉的)
            with open(allowlist_file, 'r', encoding='utf-8') as f:
                for line in f:
                    existing_lines.append(line) # 保留原始行
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        existing_artists_in_file.add(stripped_line)
                    elif stripped_line.startswith('#') and len(stripped_line) > 1: # 處理註解掉的作者
                        # 嘗試從註解中提取作者名
                        # 這裡的邏輯需要更穩健，但目前只是做存在性檢查
                        artist_from_comment = stripped_line.lstrip('#').strip().split(' ')[0] # 假設作者名在最前面
                        if artist_from_comment:
                            existing_artists_in_file.add(artist_from_comment)

            # 找出真正的新作者 (不在 existing_artists_in_file 集合中的)
            for artist in all_artists_in_dir:
                if artist not in existing_artists_in_file:
                    new_artists.append(artist)
            
            if new_artists:
                print(f"\n[更新] 發現 {len(new_artists)} 位新作者，正在加入白名單...")
                with open(allowlist_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n# --- [{time.strftime('%Y-%m-%d %H:%M')}] 新增作者 ---\n")
                    for artist in new_artists:
                        f.write(f"{artist}\n") # 預設不加 #，即全選
                
                print("="*60)
                print(f"發現新作者並已加入清單 ({allowlist_file})。")
                print("請去確認一下哪些要訓練，哪些是不要訓練。若要略過某位作者，請在行首加上井字號 (#)。") # 修正說明文字
                print("="*60)
                input("編輯完成並存檔後，請按 Enter 鍵繼續程式...")
            else:
                print(f"\n白名單檢查完畢，無新作者。")
                
        except Exception as e:
            print(f"讀取或更新白名單失敗: {e}")
            return []

    # 3. 讀取最終確認後的名單
    valid_artists = []
    try:
        with open(allowlist_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳過空行和註解
                if line and not line.startswith('#'):
                    # 簡單驗證該目錄是否真的存在於原始資料中
                    if line in all_artists_in_dir:
                        valid_artists.append(line)
                    else:
                        # 可能是舊的資料夾被刪除了，或者打錯字
                        # print(f"警告: 白名單中的 '{line}' 在來源目錄中找不到，已忽略。")
                        pass
    except Exception as e:
        print(f"讀取白名單失敗: {e}")
        return []

    print(f"\n已確認選取 {len(valid_artists)} 位作者進行處理。")
    return valid_artists

def check_disk_space_and_confirm(target_dir, artists, trained_history, num_samples_per_artist, original_data_dir):
    """
    預估空間並檢查磁碟剩餘容量
    """
    # 1. 取得磁碟剩餘空間
    drive = os.path.splitdrive(os.path.abspath(target_dir))[0]
    if not drive: drive = '.' # Linux/Mac
    
    try:
        usage = psutil.disk_usage(drive)
        free_gb = usage.free / (1024**3)
    except Exception:
        free_gb = 0
        print("無法取得磁碟空間資訊，跳過檢查。")
        return True

    # 2. 預估所需空間
    estimated_total_mb = 0
    artist_stats = {'full': 0, 'reduced': 0}
    
    if num_samples_per_artist is not None: # 如果設定了目標張數
        AVG_IMG_SIZE_MB_PER_SAMPLE = 1.2 # 上調至 1.2MB (保守估計高畫質漫畫)
        for artist in artists:
            is_trained = artist in trained_history
            if is_trained:
                count = int(num_samples_per_artist * 0.2)
                artist_stats['reduced'] += 1
            else:
                count = num_samples_per_artist
                artist_stats['full'] += 1
            estimated_total_mb += count * AVG_IMG_SIZE_MB_PER_SAMPLE
    else: # 如果沒有設定目標張數，則根據原始資料大小來估計
        # 遍歷原始資料目錄，計算所有 zip 和圖片檔案的總大小
        raw_data_size_mb = 0
        for artist_name in artists:
            artist_src_path = os.path.join(original_data_dir, artist_name)
            if os.path.exists(artist_src_path):
                for root, dirs, files in os.walk(artist_src_path):
                    for file in files:
                        # 估計壓縮檔或圖片的大小
                        if file.lower().endswith(tuple(['.zip', '.jpg', '.jpeg', '.png', '.bmp', '.webp'])):
                            raw_data_size_mb += os.path.getsize(os.path.join(root, file)) / (1024**2)

        # 考慮解壓縮和處理後的膨脹 (上調至 1.5 倍以確保安全空間)
        estimated_total_mb = raw_data_size_mb * 1.5
        artist_stats['full'] = len(artists) # 視為所有都是全量處理

    estimated_gb = estimated_total_mb / 1024
    
    print("\n" + "="*40)
    print("硬碟空間評估")
    print("="*40)
    print(f"全量訓練作者數 (新資料): {artist_stats['full']}")
    print(f"減量訓練作者數 (已訓練): {artist_stats['reduced']}")
    print(f"預估目標資料集大小: 約 {estimated_gb:.2f} GB (這不含中間產物如 temp_unzip)")
    print(f"磁碟剩餘空間: {free_gb:.2f} GB")
    
    if estimated_gb > free_gb * 0.9:
        print("\n[警告] 預估資料集大小接近或超過磁碟剩餘空間，可能導致錯誤！")
    elif estimated_gb > free_gb * 0.5:
        print("\n[注意] 預估資料集大小將佔用超過一半的剩餘空間。")
    else:
        print("\n[狀態] 空間充足。")
        
    # 等待使用者確認
    while True:
        choice = input("\n是否繼續執行？(y/n): ").lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False

def prepare_dataset(
    original_data_dir,
    target_dataset_dir,
    selected_artists,
    trained_history,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    num_samples_per_artist=None,
    min_aspect_ratio=0.5,
    max_aspect_ratio=2.0,
):
    # 清理目標資料夾 (如果存在)
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.makedirs(os.path.join(target_dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dataset_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dataset_dir, 'test'), exist_ok=True)

    total_artists = len(selected_artists)
    for idx, artist_name in enumerate(selected_artists):
        print(f"[{idx+1}/{total_artists}] 正在處理: {artist_name} ...")
        
        # 決定採樣數量
        current_target_samples = num_samples_per_artist
        if num_samples_per_artist:
            if artist_name in trained_history:
                current_target_samples = int(num_samples_per_artist * 0.2)
                print(f"  -> 歷史紀錄：已訓練。取樣調整為 20% ({current_target_samples} 張)")
            else:
                print(f"  -> 歷史紀錄：新資料。全量取樣 ({current_target_samples} 張)")
        
        artist_original_dir = os.path.join(original_data_dir, artist_name)
        temp_unzip_dir = os.path.join('temp_unzip', artist_name)
        
        if os.path.exists(temp_unzip_dir):
             shutil.rmtree(temp_unzip_dir)
        os.makedirs(temp_unzip_dir, exist_ok=True)

        all_images = []
        images_by_work = defaultdict(list)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

        # 處理原始資料夾
        if os.path.exists(artist_original_dir):
            for item in os.listdir(artist_original_dir):
                item_path = os.path.join(artist_original_dir, item)
                if item.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(item_path, 'r') as zip_ref:
                            image_files_in_zip = [
                                f for f in zip_ref.namelist() 
                                if not f.startswith('__MACOSX') and f.lower().endswith(valid_extensions)
                            ]
                            zip_ref.extractall(temp_unzip_dir)
                            for img_file in image_files_in_zip:
                                extracted_path = os.path.join(temp_unzip_dir, img_file)
                                if os.path.exists(extracted_path) and is_image_valid(extracted_path, min_aspect_ratio, max_aspect_ratio):
                                    images_by_work[item].append(extracted_path)
                    except zipfile.BadZipFile:
                        print(f"  警告：無法開啟 ZIP 檔案 {item_path}。")
                elif os.path.isdir(item_path):
                    for img_name in os.listdir(item_path):
                        full_img_path = os.path.join(item_path, img_name)
                        if img_name.lower().endswith(valid_extensions) and is_image_valid(full_img_path, min_aspect_ratio, max_aspect_ratio):
                            images_by_work[item].append(full_img_path)
                elif item.lower().endswith(valid_extensions):
                    if is_image_valid(item_path, min_aspect_ratio, max_aspect_ratio):
                        images_by_work['__loose_files__'].append(item_path)
        else:
            print(f"  警告: 原始目錄不存在，跳過。")
            continue

        # 排除頭尾
        for work_name, work_images in images_by_work.items():
            if len(work_images) > 4:
                work_images.sort()
                all_images.extend(work_images[1:-3])
            elif len(work_images) > 0:
                all_images.extend(work_images)

        random.shuffle(all_images)

        # 取樣
        if current_target_samples is not None and len(all_images) > current_target_samples:
            all_images = random.sample(all_images, current_target_samples)
        
        if len(all_images) < 3:
            print(f"  警告：有效圖片數量不足 ({len(all_images)}張)，跳過。")
            if os.path.exists(temp_unzip_dir):
                shutil.rmtree(temp_unzip_dir)
            continue

        # 分割
        num_total = len(all_images)
        num_train = int(num_total * train_split)
        num_val = int(num_total * val_split)
        
        train_images = all_images[:num_train]
        val_images = all_images[num_train : num_train + num_val]
        test_images = all_images[num_train + num_val:]

        # 複製
        for split, imgs in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            dest_dir = os.path.join(target_dataset_dir, split, artist_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in imgs:
                dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    dest_path = os.path.join(dest_dir, f"{base}_{random.randint(1000, 9999)}{ext}")
                shutil.copy(img_path, dest_path)

        print(f"  完成。 (Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)})")
        
        if os.path.exists(temp_unzip_dir):
            shutil.rmtree(temp_unzip_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='準備漫畫家畫風分類資料集 (支援互動式白名單與增量訓練)')
    parser.add_argument('--original_data_dir', type=str, default='MangaOriginalData', help='原始資料來源目錄')
    parser.add_argument('--target_dataset_dir', type=str, default='Manga_Dataset_Clean', help='目標資料集目錄')
    parser.add_argument('--artists', nargs='*', default=[], help='(可選) 直接指定要處理的藝術家，若使用此參數將跳過白名單流程')
    parser.add_argument('--whitelist', type=str, default='whitelist.txt', help='白名單檔案名稱')
    parser.add_argument('--history', type=str, default='trained_history.txt', help='訓練歷史紀錄檔案')
    parser.add_argument('--num_samples_per_artist', type=int, default=None, help='每個藝術家取樣的總張數')
    parser.add_argument('--min_aspect_ratio', type=float, default=0.5, help='最小長寬比')
    parser.add_argument('--max_aspect_ratio', type=float, default=2.0, help='最大長寬比')
    
    args = parser.parse_args()

    selected_artists = args.artists
    trained_history = load_trained_history(args.history)
    
    if not selected_artists:
        # 1. 啟用互動式白名單流程
        selected_artists = get_user_confirmed_allowlist(args.original_data_dir, args.whitelist)
        
        if not selected_artists:
            print("沒有選擇任何藝術家或白名單為空，程式結束。")
            exit()
            
        # 2. 硬碟空間評估與確認
        if not check_disk_space_and_confirm(args.target_dataset_dir, selected_artists, trained_history, args.num_samples_per_artist, args.original_data_dir):
            print("使用者取消操作。")
            exit()

    # 3. 執行處理
    prepare_dataset(
        args.original_data_dir, 
        args.target_dataset_dir, 
        selected_artists,
        trained_history,
        num_samples_per_artist=args.num_samples_per_artist,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
    )
    print("資料集準備完成！")