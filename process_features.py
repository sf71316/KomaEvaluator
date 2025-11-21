import argparse
import subprocess
import os
import sys

def run_command(cmd):
    """執行 shell 指令並即時顯示輸出"""
    print(f"\n[執行指令] {' '.join(cmd)}")
    try:
        # 使用 subprocess.run 執行，並讓 stdout/stderr 直接輸出到終端
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"指令執行失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n使用者中斷操作。")
        return False

def main():
    parser = argparse.ArgumentParser(description='漫畫特徵提取與合併流水線 (一鍵完成)')
    parser.add_argument('--src_dir', type=str, default='Manga_Dataset_Clean', help='清洗後的資料集目錄 (輸入)')
    parser.add_argument('--faces_dir', type=str, default='Intermediate_Faces', help='人臉特徵輸出目錄')
    parser.add_argument('--patches_dir', type=str, default='Intermediate_Patches', help='紋理特徵輸出目錄')
    parser.add_argument('--output_dir', type=str, default='Manga_Dataset_Mixed', help='最終合併資料集目錄 (輸出)')
    
    parser.add_argument('--num_workers', type=str, default='8', help='並行處理的核心數')
    parser.add_argument('--target_count', type=str, default='400', help='每個畫師的目標特徵數量')
    parser.add_argument('--patch_size', type=str, default='224', help='紋理切塊大小')
    parser.add_argument('--cascade', type=str, default='lbpcascade_animeface.xml', help='OpenCV Cascade 檔案')
    
    parser.add_argument('--skip_faces', action='store_true', help='跳過人臉裁切步驟')
    parser.add_argument('--skip_patches', action='store_true', help='跳過紋理提取步驟')
    
    args = parser.parse_args()

    python_exe = sys.executable

    # 1. 人臉裁切 (Crop Faces)
    if not args.skip_faces:
        print("="*60)
        print("步驟 1/3: 人臉裁切 (Face Cropping)")
        print("="*60)
        cmd_faces = [
            python_exe, 'crop_faces.py',
            '--src_dir', args.src_dir,
            '--dst_dir', args.faces_dir,
            '--num_workers', args.num_workers,
            '--cascade', args.cascade,
            '--target_count', args.target_count
        ]
        if not run_command(cmd_faces):
            print("人臉裁切失敗，流程終止。")
            sys.exit(1)

    # 2. 紋理提取 (Prepare Patches)
    if not args.skip_patches:
        print("\n" + "="*60)
        print("步驟 2/3: 紋理提取 (Texture Patch Extraction)")
        print("="*60)
        cmd_patches = [
            python_exe, 'prepare_patches.py',
            '--src_dir', args.src_dir,
            '--dst_dir', args.patches_dir,
            '--num_workers', args.num_workers,
            '--target_count', args.target_count,
            '--patch_size', args.patch_size
        ]
        if not run_command(cmd_patches):
            print("紋理提取失敗，流程終止。")
            sys.exit(1)

    # 3. 合併資料集 (Merge and Split)
    print("\n" + "="*60)
    print("步驟 3/3: 資料集合併 (Merging Datasets)")
    print("="*60)
    
    # 根據前面的步驟決定要合併哪些目錄
    dirs_to_merge = []
    if not args.skip_faces and os.path.exists(args.faces_dir):
        dirs_to_merge.append(args.faces_dir)
    if not args.skip_patches and os.path.exists(args.patches_dir):
        dirs_to_merge.append(args.patches_dir)
        
    # 如果使用者跳過了某些步驟，但目錄其實存在，也應該允許合併嗎？
    # 這裡假設使用者如果沒跳過，就一定想合併新產生的。如果跳過了，但想合併舊的，手動檢查目錄是否存在。
    if args.skip_faces and os.path.exists(args.faces_dir):
         dirs_to_merge.append(args.faces_dir)
    if args.skip_patches and os.path.exists(args.patches_dir):
         dirs_to_merge.append(args.patches_dir)
    
    # 去重
    dirs_to_merge = list(set(dirs_to_merge))

    if not dirs_to_merge:
        print("沒有可合併的來源目錄，流程結束。")
        sys.exit(0)

    cmd_merge = [
        python_exe, 'merge_and_split.py',
        '--dirs', *dirs_to_merge,
        '--dst_dir', args.output_dir,
        '--val_ratio', '0.15',
        '--test_ratio', '0.15'
    ]
    
    if not run_command(cmd_merge):
        print("資料集合併失敗。")
        sys.exit(1)

    print("\n" + "="*60)
    print("特徵處理流程完成！")
    print(f"最終資料集位於: {args.output_dir}")
    print("接下來，請執行訓練指令：")
    print(f"python train.py --data_dir {args.output_dir} --model convnext_v2_tiny_local --epochs 50 --record_history")
    print("="*60)

if __name__ == '__main__':
    main()
