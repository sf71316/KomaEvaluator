import argparse
import subprocess
import os
import sys

def run_command(cmd, env=None):
    """執行 shell 指令並即時顯示輸出"""
    print(f"\n[執行指令] {' '.join(cmd)}")
    try:
        # 使用 subprocess.run 執行，並讓 stdout/stderr 直接輸出到終端
        result = subprocess.run(cmd, check=True, env=env)
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
    
    parser.add_argument('--num_workers', type=str, default='4', help='並行處理的核心數')
    parser.add_argument('--target_count', type=str, default='400', help='每個畫師的目標特徵數量')
    parser.add_argument('--patch_size', type=str, default='224', help='紋理切塊大小')
    parser.add_argument('--cascade', type=str, default='lbpcascade_animeface.xml', help='OpenCV Cascade 檔案')
    
    parser.add_argument('--skip_faces', action='store_true', help='跳過人臉裁切步驟')
    parser.add_argument('--skip_patches', action='store_true', help='跳過紋理提取步驟')
    parser.add_argument('--low_priority', action='store_true', help='調降子進程優先度 (背景運行)')
    
    # 新增一個參數來控制警告抑制，預設為關閉，但可以啟用
    parser.add_argument('--suppress_libpng_warnings', action='store_true', help='抑制 libpng 的 iCCP 警告')
    
    args = parser.parse_args()

    python_exe = sys.executable

    # 為子進程準備環境變數
    my_env = os.environ.copy()
    if args.suppress_libpng_warnings:
        my_env["PNG_UNTRUSTED_ICC"] = "0"
        print("[警告抑制] libpng 的 iCCP 警告已抑制。")

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
            '--cascade', args.cascade
        ]
        if args.low_priority:
            cmd_faces.append('--low_priority')

        if not run_command(cmd_faces, env=my_env):
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
        if args.low_priority:
            cmd_patches.append('--low_priority')

        if not run_command(cmd_patches, env=my_env):
            print("紋理提取失敗，流程終止。")
            sys.exit(1)

    # 3. 合併資料集 (Merge and Split)
    print("\n" + "="*60)
    print("步驟 3/3: 資料集合併 (Merging Datasets)")
    print("="*60)
    
    dirs_to_merge = []
    if not args.skip_faces and os.path.exists(args.faces_dir):
        dirs_to_merge.append(args.faces_dir)
    if not args.skip_patches and os.path.exists(args.patches_dir):
        dirs_to_merge.append(args.patches_dir)
        
    if args.skip_faces and os.path.exists(args.faces_dir):
         dirs_to_merge.append(args.faces_dir)
    if args.skip_patches and os.path.exists(args.patches_dir):
         dirs_to_merge.append(args.patches_dir)
    
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
    if args.low_priority:
        cmd_merge.append('--low_priority')

    if not run_command(cmd_merge, env=my_env):
        print("資料集合併失敗。")
        sys.exit(1)

    print("\n" + "="*60)
    print("特徵處理流程完成！")
    print(f"最終資料集位於: {args.output_dir}")
    print("接下來，請執行訓練指令：")
    
    possible_checkpoint = os.path.join('DL_Output_Models', 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'checkpoint_last.pth')
    possible_model = os.path.join('DL_Output_Models', 'convnextv2_tiny.fcmae_ft_in22k_in1k', 'final_model.pth')
    
    if os.path.exists(possible_checkpoint):
        print("\n[偵測到上次訓練的 Checkpoint]")
        print("若要【接續訓練】(架構/類別未變)，請使用：")
        print(f"python train.py --data_dir {args.output_dir} --model convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 50 --record_history --resume_path {possible_checkpoint}")
        print("\n若這是【增量訓練】(有新增作者/類別)，請使用：")
        print(f"python train.py --data_dir {args.output_dir} --model convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 50 --record_history --load_path {possible_checkpoint}")
    elif os.path.exists(possible_model):
        print("\n[偵測到舊模型權重]")
        print("若要基於舊模型進行【微調】(Fine-tuning)，請使用：")
        print(f"python train.py --data_dir {args.output_dir} --model convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 50 --record_history --load_path {possible_model}")
    else:
        print("\n[全新訓練]")
        print(f"python train.py --data_dir {args.output_dir} --model convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 50 --record_history")
        
    print("="*60)

if __name__ == '__main__':
    main()
