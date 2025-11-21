import os
import zipfile
from collections import defaultdict

def analyze_artists(data_dir):
    artist_stats = defaultdict(int)
    for artist_name in os.listdir(data_dir):
        artist_dir = os.path.join(data_dir, artist_name)
        if os.path.isdir(artist_dir):
            for item in os.listdir(artist_dir):
                item_path = os.path.join(artist_dir, item)
                if item.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(item_path, 'r') as zip_ref:
                            # 過濾掉非圖片檔案
                            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                            artist_stats[artist_name] += len(image_files)
                    except zipfile.BadZipFile:
                        print(f"警告：無法開啟 ZIP 檔案 {item_path}，可能已損壞。")
                elif os.path.isdir(item_path):
                    # 過濾掉非圖片檔案
                    image_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                    artist_stats[artist_name] += len(image_files)
    return artist_stats

if __name__ == '__main__':
    data_dir = 'MangaOriginalData'
    artist_stats = analyze_artists(data_dir)
    
    print("藝術家圖片數量統計：")
    for artist, count in sorted(artist_stats.items(), key=lambda item: item[1], reverse=True):
        print(f"{artist}: {count} 張")
