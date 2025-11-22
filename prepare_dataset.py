import os
import zipfile
import shutil
import random
from collections import defaultdict
import argparse
from PIL import Image
import psutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import py7zr

def is_image_valid(image_path, min_aspect_ratio, max_aspect_ratio):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if height == 0: return False, "Height is 0"
            aspect_ratio = width / height
            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                return False, f"Aspect ratio {aspect_ratio:.2f} out of range"
            return True, "OK"
    except Exception as e:
        return False, f"Open error: {str(e)}"

def load_trained_history(history_file='trained_history.txt'):
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
    if not os.path.exists(src_dir):
        print(f"錯誤: 來源目錄 {src_dir} 不存在。")
        return []

    all_artists_in_dir = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    if not all_artists_in_dir:
        print("錯誤: 來源目錄下沒有發現任何資料夾。")
        return []

    new_artists = []
    
    if not os.path.exists(allowlist_file):
        print(f"\n[初始化] 未發現白名單檔案，正在建立 {allowlist_file} ...")
        new_artists = all_artists_in_dir
        try:
            with open(allowlist_file, 'w', encoding='utf-8') as f:
                f.write("# ==========================================\n")
                f.write("# 訓練白名單設定檔\n")
                f.write("# 請保留您想要處理的作者資料夾名稱\n")
                f.write("# 若要略過某位作者，請在行首加上井字號 (#)\n")
                f.write("# ==========================================\n\n")
                for artist in new_artists:
                    f.write(f"{artist}\n")
            print(f"已成功建立預設白名單: {os.path.abspath(allowlist_file)}")
        except IOError as e:
            print(f"無法寫入白名單檔案: {e}")
            return []
        print("="*60)
        print(f"我幫你建了一份白名單 ({allowlist_file})，你去看一下哪些要訓練，哪些是不要訓練。")
        print("="*60)
        input("編輯完成並存檔後，請按 Enter 鍵繼續程式...")
        
    else:
        try:
            existing_artists_in_file = set()
            with open(allowlist_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            for artist in all_artists_in_dir:
                if artist not in raw_content: 
                     new_artists.append(artist)
            
            if new_artists:
                print(f"\n[更新] 發現 {len(new_artists)} 位新作者，正在加入白名單...")
                with open(allowlist_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n# --- [{time.strftime('%Y-%m-%d %H:%M')}] 新增作者 ---\n")
                    for artist in new_artists:
                        f.write(f"{artist}\n")
                print("="*60)
                print(f"發現新作者並已加入清單 ({allowlist_file})。")
                print("請去確認一下哪些要訓練，哪些是不要訓練。若要略過某位作者，請在行首加上井字號 (#)。")
                print("="*60)
                input("編輯完成並存檔後，請按 Enter 鍵繼續程式...")
            else:
                print(f"\n白名單檢查完畢，無新作者。")
        except Exception as e:
            print(f"讀取或更新白名單失敗: {e}")
            return []

    valid_artists = []
    try:
        with open(allowlist_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line in all_artists_in_dir:
                        valid_artists.append(line)
    except Exception as e:
        print(f"讀取白名單失敗: {e}")
        return []

    print(f"\n已確認選取 {len(valid_artists)} 位作者進行處理。")
    return valid_artists

def check_disk_space_and_confirm(target_dir, artists, trained_history, num_samples_per_artist, original_data_dir):
    drive = os.path.splitdrive(os.path.abspath(target_dir))[0]
    if not drive: drive = '.'
    
    try:
        usage = psutil.disk_usage(drive)
        free_gb = usage.free / (1024**3)
    except Exception:
        free_gb = 0
        print("無法取得磁碟空間資訊，跳過檢查。")
        return True

    estimated_total_mb = 0
    artist_stats = {'full': 0, 'reduced': 0}
    
    if num_samples_per_artist is not None:
        AVG_IMG_SIZE_MB_PER_SAMPLE = 1.2
        for artist in artists:
            is_trained = artist in trained_history
            if is_trained:
                count = int(num_samples_per_artist * 0.2)
                artist_stats['reduced'] += 1
            else:
                count = num_samples_per_artist
                artist_stats['full'] += 1
            estimated_total_mb += count * AVG_IMG_SIZE_MB_PER_SAMPLE
    else:
        raw_data_size_mb = 0
        for artist_name in artists:
            artist_src_path = os.path.join(original_data_dir, artist_name)
            if os.path.exists(artist_src_path):
                for root, dirs, files in os.walk(artist_src_path):
                    for file in files:
                        if file.lower().endswith(tuple(['.zip', '.7z', '.jpg', '.jpeg', '.png', '.bmp', '.webp'])):
                            raw_data_size_mb += os.path.getsize(os.path.join(root, file)) / (1024**2)
        estimated_total_mb = raw_data_size_mb * 1.3
        artist_stats['full'] = len(artists)

    estimated_gb = estimated_total_mb / 1024
    
    print("\n" + "="*40)
    print("硬碟空間評估")
    print("="*40)
    print(f"全量訓練作者數 (新資料): {artist_stats['full']}")
    print(f"減量訓練作者數 (已訓練): {artist_stats['reduced']}")
    print(f"預估目標資料集大小: 約 {estimated_gb:.2f} GB")
    print(f"磁碟剩餘空間: {free_gb:.2f} GB")
    
    if estimated_gb > free_gb * 0.9:
        print("\n[警告] 預估資料集大小接近或超過磁碟剩餘空間，可能導致錯誤！")
    elif estimated_gb > free_gb * 0.5:
        print("\n[注意] 預估資料集大小將佔用超過一半的剩餘空間。")
    else:
        print("\n[狀態] 空間充足。")
        
    while True:
        choice = input("\n是否繼續執行？(y/n): ").lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False

def copy_file_task(src, dst):
    try:
        shutil.copy(src, dst)
        return True
    except Exception as e:
        print(f"複製失敗 {src}: {e}")
        return False

def extract_and_scan_archive(archive_path, extract_to, valid_extensions, min_ar, max_ar, debug=False):
    """
    解壓縮並掃描圖片，返回有效圖片路徑列表。
    支援 .zip 和 .7z。 .rar 不支援但會提示。
    """
    valid_images = []
    failed_images = [] # (filename, reason)
    total_files_count = 0
    
    ext = os.path.splitext(archive_path)[1].lower()
    
    if ext == '.rar':
        print(f"  [跳過] 不支援 RAR 格式: {os.path.basename(archive_path)} (請手動解壓或轉為 zip/7z)")
        return []

    try:
        extracted_files = []
        
        if ext == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                total_files_count = len(all_files)
                # 過濾掉 __MACOSX 等系統檔
                files_to_extract = [f for f in all_files if not f.startswith('__MACOSX')]
                zip_ref.extractall(extract_to, members=files_to_extract)
                extracted_files = files_to_extract
                
        elif ext == '.7z':
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                all_files = z.getnames()
                total_files_count = len(all_files)
                z.extractall(path=extract_to)
                extracted_files = all_files
        else:
            return [] # Should not happen if caller checks extension

        # 掃描解壓後的檔案
        for file_rel_path in extracted_files:
            full_path = os.path.join(extract_to, file_rel_path)
            if os.path.isfile(full_path) and file_rel_path.lower().endswith(valid_extensions):
                is_valid, reason = is_image_valid(full_path, min_ar, max_ar)
                if is_valid:
                    valid_images.append(full_path)
                else:
                    if debug:
                        failed_images.append((file_rel_path, reason))
            
        if debug:
            print(f"  [Debug] 壓縮檔: {os.path.basename(archive_path)}")
            print(f"    - 檔案總數: {total_files_count}")
            print(f"    - 有效圖片: {len(valid_images)}")
            print(f"    - 驗證失敗: {len(failed_images)}")
            if failed_images:
                print(f"    - 失敗詳情 (前 5 筆):")
                for fail_item in failed_images[:5]:
                    print(f"      * {fail_item[0]}: {fail_item[1]}")
                if len(failed_images) > 5:
                    print(f"      ... 還有 {len(failed_images) - 5} 筆")

    except (zipfile.BadZipFile, py7zr.exceptions.Bad7zFile):
        print(f"  [錯誤] 損壞的壓縮檔: {os.path.basename(archive_path)}")
    except Exception as e:
        print(f"  [錯誤] 解壓失敗 {os.path.basename(archive_path)}: {e}")

    return valid_images

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
    num_workers=8,
    debug_mode=False
):
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.makedirs(os.path.join(target_dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dataset_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dataset_dir, 'test'), exist_ok=True)

    total_artists = len(selected_artists)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    
    for idx, artist_name in enumerate(selected_artists):
        print(f"[{idx+1}/{total_artists}] 正在處理: {artist_name} ...")
        
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

        if os.path.exists(artist_original_dir):
            for item in os.listdir(artist_original_dir):
                item_path = os.path.join(artist_original_dir, item)
                
                # 處理壓縮檔
                if item.lower().endswith(('.zip', '.7z', '.rar')):
                    extracted_imgs = extract_and_scan_archive(
                        item_path, 
                        temp_unzip_dir, 
                        valid_extensions, 
                        min_aspect_ratio, 
                        max_aspect_ratio, 
                        debug=debug_mode
                    )
                    if extracted_imgs:
                        images_by_work[item].extend(extracted_imgs)
                
                # 處理已解壓資料夾
                elif os.path.isdir(item_path):
                    folder_imgs = []
                    failed_in_folder = []
                    for img_name in os.listdir(item_path):
                        if img_name.lower().endswith(valid_extensions):
                            full_img_path = os.path.join(item_path, img_name)
                            is_valid, reason = is_image_valid(full_img_path, min_aspect_ratio, max_aspect_ratio)
                            if is_valid:
                                folder_imgs.append(full_img_path)
                            elif debug_mode:
                                failed_in_folder.append((img_name, reason))
                    
                    if folder_imgs:
                        images_by_work[item].extend(folder_imgs)
                    
                    if debug_mode and failed_in_folder:
                         print(f"  [Debug] 資料夾: {item}")
                         print(f"    - 驗證失敗: {len(failed_in_folder)}")
                         # (可選) 顯示資料夾內的失敗詳情

                # 處理散落的圖片
                elif item.lower().endswith(valid_extensions):
                    is_valid, reason = is_image_valid(item_path, min_aspect_ratio, max_aspect_ratio)
                    if is_valid:
                        images_by_work['__loose_files__'].append(item_path)
                    elif debug_mode:
                        print(f"  [Debug] 散圖失敗 {item}: {reason}")
        else:
            print(f"  警告: 原始目錄不存在，跳過。")
            continue

        for work_name, work_images in images_by_work.items():
            # 去頭去尾 (針對漫畫章節簡單過濾封面/版權頁)
            if len(work_images) > 4:
                work_images.sort()
                all_images.extend(work_images[1:-3])
            elif len(work_images) > 0:
                all_images.extend(work_images)

        random.shuffle(all_images)

        if current_target_samples is not None:
             low_threshold = current_target_samples * 0.5
             if len(all_images) < low_threshold:
                 print(f"  [警告] 圖片數量 ({len(all_images)}) 低於目標的 50% ({low_threshold})！可能影響訓練效果。")

        if current_target_samples is not None and len(all_images) > current_target_samples:
            all_images = random.sample(all_images, current_target_samples)
        
        if len(all_images) < 3:
            print(f"  警告：有效圖片數量不足 ({len(all_images)}張)，跳過。")
            if os.path.exists(temp_unzip_dir):
                shutil.rmtree(temp_unzip_dir)
            continue

        num_total = len(all_images)
        num_train = int(num_total * train_split)
        num_val = int(num_total * val_split)
        
        train_images = all_images[:num_train]
        val_images = all_images[num_train : num_train + num_val]
        test_images = all_images[num_train + num_val:]

        copy_tasks = []
        for split, imgs in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            dest_dir = os.path.join(target_dataset_dir, split, artist_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in imgs:
                dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    dest_path = os.path.join(dest_dir, f"{base}_{random.randint(1000, 9999)}{ext}")
                copy_tasks.append((img_path, dest_path))
        
        if copy_tasks:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(copy_file_task, src, dst) for src, dst in copy_tasks]
                for _ in as_completed(futures):
                    pass
        
        print(f"  完成。 (Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)})")
        
        if os.path.exists(temp_unzip_dir):
            shutil.rmtree(temp_unzip_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='準備漫畫家畫風分類資料集')
    parser.add_argument('--original_data_dir', type=str, default='MangaOriginalData', help='原始資料來源目錄')
    parser.add_argument('--target_dataset_dir', type=str, default='Manga_Dataset_Clean', help='目標資料集目錄')
    parser.add_argument('--artists', nargs='*', default=[], help='(可選) 直接指定要處理的藝術家')
    parser.add_argument('--whitelist', type=str, default='whitelist.txt', help='白名單檔案名稱')
    parser.add_argument('--history', type=str, default='trained_history.txt', help='訓練歷史紀錄檔案')
    parser.add_argument('--num_samples_per_artist', type=int, default=None, help='每個藝術家取樣的總張數')
    parser.add_argument('--min_aspect_ratio', type=float, default=0.5, help='最小長寬比')
    parser.add_argument('--max_aspect_ratio', type=float, default=2.0, help='最大長寬比')
    parser.add_argument('--num_workers', type=int, default=16, help='複製檔案的執行緒數量')
    parser.add_argument('--debug', action='store_true', help='開啟 Debug 模式，顯示詳細解壓縮與驗證資訊') # 新增參數
    
    args = parser.parse_args()

    selected_artists = args.artists
    trained_history = load_trained_history(args.history)
    
    if not selected_artists:
        selected_artists = get_user_confirmed_allowlist(args.original_data_dir, args.whitelist)
        
        if not selected_artists:
            print("沒有選擇任何藝術家或白名單為空，程式結束。")
            exit()
            
        if not check_disk_space_and_confirm(args.target_dataset_dir, selected_artists, trained_history, args.num_samples_per_artist, args.original_data_dir):
            print("使用者取消操作。")
            exit()

    prepare_dataset(
        args.original_data_dir, 
        args.target_dataset_dir, 
        selected_artists,
        trained_history,
        num_samples_per_artist=args.num_samples_per_artist,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        num_workers=args.num_workers,
        debug_mode=args.debug # 傳遞 debug 參數
    )
    print("\n" + "="*60)
    print("資料準備完成！")
    print("接下來，請執行特徵提取與合併流程：")
    print(f"python process_features.py --src_dir {args.target_dataset_dir}")
    print("="*60)