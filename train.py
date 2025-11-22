import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import sys
import numpy as np
import json
import time
import argparse
import copy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import shutil
import timm 
from timm.utils import ModelEmaV2

# ==============================================================================
# Helper Functions
# ==============================================================================

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint_last.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        shutil.copyfile(last_path, best_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler=None, ema_model=None, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: Checkpoint {checkpoint_path} 不存在")
        return 0, 0.0, float('inf')

    print(f"正在載入 Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    # Remove module. prefix if needed
    if list(state_dict.keys())[0].startswith('module.') and not list(model.state_dict().keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    # Handle shape mismatch (e.g., for incremental training with new classes)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"  警告: 跳過載入層 '{k}'，因為形狀不匹配 (Checkpoint: {v.shape}, Model: {model_state_dict[k].shape})")
        else:
             pass # Skip missing keys
             
    model.load_state_dict(filtered_state_dict, strict=False)

    # Only load optimizer if fully matched
    if len(filtered_state_dict) == len(state_dict):
        print("  完整恢復 Optimizer 和 Scheduler 狀態")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            print(f"  警告: 載入優化器狀態失敗 ({e})，將使用新初始化的優化器。")
    else:
        print("  警告: 由於模型結構改變，跳過載入 Optimizer/Scheduler/Scaler 狀態。")

    if ema_model and 'ema_model_state_dict' in checkpoint:
        try:
             ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
        except:
             print("  警告: 無法完全恢復 EMA 模型狀態")
        
    epoch = checkpoint['epoch']
    best_acc = checkpoint.get('best_acc', 0.0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    print(f"Checkpoint 載入完成 (Epoch {epoch}, Best Acc: {best_acc:.4f})")
    return epoch, best_acc, best_loss

def update_trained_history(history_file, class_names):
    print(f"正在更新訓練歷史紀錄至 {history_file} ...")
    existing_history = set()
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        existing_history.add(line)
        except Exception as e:
            print(f"讀取歷史檔案失敗: {e}")

    new_classes = [c for c in class_names if c not in existing_history]
    if new_classes:
        try:
            with open(history_file, 'a', encoding='utf-8') as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n# --- [{timestamp}] Training Session ---\n")
                for c in new_classes:
                    f.write(f"{c}\n")
            print(f"已新增 {len(new_classes)} 位作者到歷史紀錄。 ")
        except Exception as e:
            print(f"寫入歷史檔案失敗: {e}")
    else:
        print("沒有新的作者需要記錄。 ")

def build_resume_command(checkpoint_path):
    args = sys.argv[:]
    cmd = ["python"] + args
    if '--resume_path' in cmd:
        idx = cmd.index('--resume_path')
        cmd[idx + 1] = checkpoint_path
    else:
        cmd.extend(['--resume_path', checkpoint_path])
    return " ".join(cmd)

# ==============================================================================
# Augmentation Logic (Mixup/Cutmix)
# ==============================================================================

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ==============================================================================
# Training Loop
# ==============================================================================

def train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, writer, args, start_epoch=0, best_acc=0.0, best_loss=float('inf'), scaler=None, ema_model=None, device='cuda:0'):
    since = time.time()
    epochs = args.epochs
    accumulation_steps = args.accumulation_steps
    early_stopping_patience = args.early_stopping_patience
    early_stopping_delta = args.early_stopping_delta
    use_mixup = args.use_mixup
    mixup_alpha = args.mixup_alpha
    use_cutmix = args.use_cutmix
    cutmix_alpha = args.cutmix_alpha
    use_amp = args.amp
    use_bf16 = args.bf16
    checkpoint_dir = os.path.dirname(args.save_path)
    if not checkpoint_dir: checkpoint_dir = '.'

    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        for epoch in range(start_epoch, epochs):
            print(f'Epoch {epoch}/{epochs - 1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_corrects_top5 = 0
                
                optimizer.zero_grad(set_to_none=True)
                eval_model = ema_model.module if ema_model is not None and phase == 'val' else model

                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    mixed = False
                    lam = 1.0
                    if phase == 'train':
                        if use_mixup and use_cutmix:
                            if torch.rand(1).item() > 0.5:
                                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                                mixed = True
                            else:
                                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device)
                                mixed = True
                        elif use_mixup:
                            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                            mixed = True
                        elif use_cutmix:
                            inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_alpha, device)
                            mixed = True

                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
                            outputs = eval_model(inputs) if phase == 'val' else model(inputs)
                            
                            if mixed:
                                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                            else:
                                loss = criterion(outputs, labels)
                                
                            if phase == 'train':
                                loss = loss / accumulation_steps

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            if (i + 1) % accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad(set_to_none=True)
                            if ema_model is not None:
                                ema_model.update(model)

                    batch_size = inputs.size(0)
                    running_loss += loss.item() * batch_size * (accumulation_steps if phase == 'train' else 1)
                    
                    if not mixed:
                        res = accuracy(outputs, labels, topk=(1, 5))
                        running_corrects += res[0].item() * batch_size / 100
                        running_corrects_top5 += res[1].item() * batch_size / 100
                    else:
                        _, preds = torch.max(outputs, 1)
                        if lam > 0.5:
                            running_corrects += torch.sum(preds == labels_a.data)
                        else:
                            running_corrects += torch.sum(preds == labels_b.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                epoch_acc_top5 = running_corrects_top5 / dataset_sizes[phase] if not (phase=='train' and (use_mixup or use_cutmix)) else 0.0

                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train_top1', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/val_top1', epoch_acc, epoch)
                    writer.add_scalar('Accuracy/val_top5', epoch_acc_top5, epoch)

                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': exp_lr_scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
                        'best_acc': best_acc,
                        'best_loss': best_loss,
                        'args': vars(args)
                    }
                    
                    is_best = False
                    if epoch_loss < best_loss - early_stopping_delta:
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        is_best = True
                    else:
                        epochs_no_improve += 1
                    
                    save_checkpoint(checkpoint, is_best, checkpoint_dir, filename=f'checkpoint_last.pth')

                if phase == 'val':
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc@1: {epoch_acc:.4f} Acc@5: {epoch_acc_top5:.4f}')
                else:
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc@1: {epoch_acc:.4f}')

            writer.add_scalar('Learning Rate', exp_lr_scheduler.get_last_lr()[0], epoch)
            exp_lr_scheduler.step()

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            print()

    except KeyboardInterrupt:
        print("\n\n[使用者中斷] 正在儲存 Checkpoint 並安全退出...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': exp_lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'ema_model_state_dict': ema_model.state_dict() if ema_model else None,
            'best_acc': best_acc,
            'best_loss': best_loss,
            'args': vars(args)
        }
        save_filename = f'checkpoint_interrupted_epoch_{epoch}.pth'
        save_checkpoint(checkpoint, False, checkpoint_dir, filename=save_filename)
        
        checkpoint_path = os.path.join(checkpoint_dir, save_filename)
        resume_cmd = build_resume_command(checkpoint_path)
        
        is_windows = os.name == 'nt'
        script_ext = 'bat' if is_windows else 'sh'
        script_name = f'resume_training.{script_ext}'
        
        with open(script_name, 'w', encoding='utf-8') as f:
            if not is_windows:
                f.write("#!/bin/bash\n")
            f.write(resume_cmd)
            f.write("\n")
            if is_windows:
                f.write("pause")
        
        if not is_windows:
            os.chmod(script_name, 0o755)

        print("="*60)
        print(f"Checkpoint 已儲存至: {checkpoint_path}")
        print(f"已建立恢復腳本: {script_name}")
        print(f"您可以直接執行 {script_name} 來恢復訓練，或是複製以下指令:")
        print(f"\n{resume_cmd}\n")
        print("="*60)
        sys.exit(0)

    time_elapsed = time.time() - since
    print(f'訓練完成，耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳驗證準確率: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='漫畫家畫風分類模型訓練 (基於 timm)')
    parser.add_argument('--data_dir', type=str, default='Manga_Dataset', help='資料集路徑')
    parser.add_argument('--model', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k', help='timm 模型名稱 (如 convnextv2_tiny.fcmae_ft_in22k_in1k)')
    parser.add_argument('--save_path', type=str, default='final_model.pth', help='儲存模型權重的檔案名稱 (相對路徑)')
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='資料載入的執行緒數量')
    parser.add_argument('--epochs', type=int, default=20, help='訓練輪數')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='標籤平滑')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='權重衰減')
    parser.add_argument('--drop_path', type=float, default=0.0, help='隨機深度率')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累積步數')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='提前停止耐心值')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001, help='提前停止最小變化')
    parser.add_argument('--use_mixup', action='store_true', help='啟用 Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    parser.add_argument('--use_cutmix', action='store_true', help='啟用 Cutmix')
    parser.add_argument('--cutmix_alpha', type=float, default=0.4, help='Cutmix alpha')
    
    # 這些參數現在由 timm 處理或簡化
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup 輪數') # 預設改為 5
    parser.add_argument('--amp', action='store_true', help='啟用 AMP 混合精度')
    parser.add_argument('--bf16', action='store_true', help='啟用 BF16')
    parser.add_argument('--model_ema', action='store_true', help='啟用 EMA')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='EMA 衰減率')

    # Checkpoint & History
    parser.add_argument('--resume_path', type=str, default=None, help='恢復訓練的 Checkpoint 路徑')
    parser.add_argument('--load_path', type=str, default=None, help='載入模型權重路徑 (如增量訓練)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='指定本地預訓練權重路徑 (若不指定則由 timm 自動下載)')
    parser.add_argument('--history_file', type=str, default='trained_history.txt', help='訓練歷史紀錄檔案')
    parser.add_argument('--record_history', action='store_true', help='訓練完成後記錄作者到歷史檔案')

    args = parser.parse_args()

    from torchvision import datasets, transforms

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 設定輸出與 Checkpoint 目錄
    if os.path.dirname(args.save_path):
        full_save_path = args.save_path
    else:
        full_save_path = os.path.join('DL_Output_Models', args.model, args.save_path)
    
    args.save_path = full_save_path
    checkpoint_dir = os.path.dirname(full_save_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始 TensorBoard (明確創建日誌目錄)
    log_dir_base = 'runs'
    os.makedirs(log_dir_base, exist_ok=True) # 確保 runs/ 根目錄存在
    log_dir_name = os.path.join(log_dir_base, f'{args.model}_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(log_dir_name, exist_ok=True) # 確保特定模型日誌目錄存在
    writer = SummaryWriter(log_dir_name)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_data_path = os.path.join(args.data_dir, 'train')
    val_data_path = os.path.join(args.data_dir, 'val')
    test_data_path = os.path.join(args.data_dir, 'test')

    image_datasets = {
        'train': datasets.ImageFolder(train_data_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_data_path, data_transforms['val']),
        'test': datasets.ImageFolder(test_data_path, data_transforms['test'])
    }
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"類別數量: {num_classes}")
    
    num_train_samples = len(image_datasets['train'])
    if num_train_samples < 1000:
        print("小型資料集模式: 自動調整超參數 (Lr, DropPath)")
        if args.lr == 0.001: args.lr = 0.00005
        if args.drop_path == 0.0: args.drop_path = 0.2
    
    # --- 模型初始化 (使用 timm) ---
    print(f"建立模型: {args.model}")
    
    # 轉換舊習慣的名稱 (e.g. convnext_v2_tiny -> convnextv2_tiny)
    timm_model_name = args.model.replace('_local', '').replace('convnext_v2_', 'convnextv2_')
    
    try:
        # 是否使用 timm 線上預訓練權重
        use_timm_pretrained = True if args.pretrained_path is None else False
        
        model_ft = timm.create_model(
            timm_model_name,
            pretrained=use_timm_pretrained,
            num_classes=num_classes,
            drop_path_rate=args.drop_path
        )
        print(f"成功使用 timm 建立模型: {timm_model_name} (Pretrained={use_timm_pretrained})")
        
        # 如果指定了本地權重，手動載入
        if args.pretrained_path:
            if os.path.exists(args.pretrained_path):
                print(f"正在載入指定預訓練權重: {args.pretrained_path}")
                checkpoint = torch.load(args.pretrained_path, map_location='cpu')
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                msg = model_ft.load_state_dict(state_dict, strict=False)
                print(f"權重載入結果: {msg}")
            else:
                print(f"警告: 找不到指定的權重檔 {args.pretrained_path}，使用隨機初始化。 ")

    except Exception as e:
        print(f"timm 建立模型失敗: {e}")
        print("請確認模型名稱是否正確 (參考 timm.list_models())")
        sys.exit(1)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 優化器 (AdamW)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA
    ema_model = None
    if args.model_ema:
        ema_model = ModelEmaV2(model_ft, decay=args.model_ema_decay, device=None)

    # 排程器
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LambdaLR, SequentialLR
        def get_warmup_lambda(current_epoch):
            return float(current_epoch) / float(max(1, args.warmup_epochs))
        warmup_scheduler = LambdaLR(optimizer_ft, lr_lambda=get_warmup_lambda)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 1e-6)
        exp_lr_scheduler = SequentialLR(optimizer_ft, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    else:
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.epochs, eta_min=args.lr * 1e-6)

    scaler = GradScaler(enabled=(args.amp and not args.bf16))

    # Resume / Load Weights
    start_epoch = 0
    best_acc = 0.0
    best_loss = float('inf')

    if args.resume_path:
        start_epoch, best_acc, best_loss = load_checkpoint(
            args.resume_path, model_ft, optimizer_ft, exp_lr_scheduler, scaler, ema_model, device
        )
        start_epoch += 1
        print(f"恢復訓練自 Epoch {start_epoch}")
    elif args.load_path:
        print(f"載入權重自: {args.load_path}")
        load_checkpoint(args.load_path, model_ft, optimizer_ft, exp_lr_scheduler, scaler, ema_model, device)
        print("權重載入完成 (僅權重)")

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False, num_workers=args.num_workers, pin_memory=True) for x in ['train', 'val', 'test']}

    # 開始訓練
    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        dataloaders, writer, args,
        start_epoch=start_epoch, best_acc=best_acc, best_loss=best_loss,
        scaler=scaler, ema_model=ema_model, device=device
    )

    # 儲存結果
    torch.save(model_ft.state_dict(), full_save_path)
    print(f"最終模型已儲存至 {full_save_path}")
    if ema_model:
        torch.save(ema_model.module.state_dict(), full_save_path.replace('.pth', '_ema.pth'))

    if args.record_history:
        update_trained_history(args.history_file, class_names)

    writer.close()

if __name__ == '__main__':
    main()
