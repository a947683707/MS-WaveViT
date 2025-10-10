import argparse
import random
import numpy as np
import datetime
import time
import os  # 添加缺失的导入
import torch
import argparse  # 添加缺失的导入
from tqdm import tqdm
from wave_models.wavehyper_solver import WaveHyperSolver

# 添加随机种子设置函数
def set_random_seeds(seed=None):
    """设置所有相关的随机种子"""
    if seed is None:
        seed = int(time.time() * 1000) % 2**32  # 使用时间戳作为种子
    
    # 确保种子在有效范围内
    seed = seed % (2**32)
    
    random.seed(seed)
    np.random.seed(seed % (2**32))  # numpy要求32位种子
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 为了更好的随机性，可以启用这些设置（会影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    return seed

# 设置 CUDA 可见设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置日志目录和文件
log_dir = r'H:\ZhangR\hyperIQA-master\log'  # 修正路径格式
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_log.txt')

# 日志写入函数
def log(msg, print_console=True):
    if print_console:
        print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

def main(config):
    start_all = time.time()  # 记录总开始时间
    
    # 设置基础随机种子
    base_seed = set_random_seeds()
    log(f'Base random seed: {base_seed}')

    # 本地数据集路径
    folder_path = {
        'ESPL_LIVE_HDR': r'H:\ZhangR\hyperIQA-master\ESPL_LIVE_HDR_Database\Images',
    }

    img_num = {
        'ESPL_LIVE_HDR': list(range(0, 1811)),
    }

    sel_num = img_num[config.dataset]
    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)
    
    # 添加全局最优模型跟踪
    global_best_srcc = 0.0
    global_best_plcc = 0.0
    global_best_round = 0
    
    # 全局最优模型保存路径
    global_save_dir = r'H:\ZhangR\hyperIQA-master\wave_models\saved_models'
    os.makedirs(global_save_dir, exist_ok=True)
    global_best_path = os.path.join(global_save_dir, 'wave_model_global_best.pth')  # 添加这行

    log('=' * 80)
    log(f'Training Configuration:')
    log(f'Dataset: {config.dataset}')
    log(f'Total Rounds: {config.train_test_num}')
    log(f'Epochs per Round: {config.epochs}')
    log(f'Batch Size: {config.batch_size}')
    log(f'Learning Rate: {config.lr}')
    log('=' * 80)
    log('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))

    # 使用tqdm显示Round进度
    round_pbar = tqdm(range(config.train_test_num), desc="Training Rounds", unit="round")
    
    # 在main函数开始处添加路径验证
    data_path = folder_path[config.dataset]
    if not os.path.exists(data_path):
        log(f'❌ Error: Dataset path does not exist: {data_path}')
        return
    
    log(f'✅ Dataset path verified: {data_path}')
    
    # 在main函数开始处预生成所有Round的种子
    round_seeds = []
    for round_idx in range(config.train_test_num):
        round_seed = (base_seed + round_idx * 1000 + hash(f"{base_seed}_{round_idx}")) % (2**32)
        round_seeds.append(round_seed)
    
    log(f'Pre-generated seeds for {config.train_test_num} rounds')
    
    # 然后在Round循环中使用
    for i in round_pbar:
        log('-' * 60)
        log(f'Round {i + 1}/{config.train_test_num} start')
        round_start = time.time()

        # 为每个Round设置不同的随机种子
        # 修正：使用确定性的种子生成，避免依赖随机状态
        round_seed = round_seeds[i]
        set_random_seeds(round_seed)
        log(f'Round {i+1} random seed: {round_seed}')
        
        # 确保每次都有不同的数据划分
        sel_num_copy = sel_num.copy()  # 创建副本避免修改原始列表
        random.shuffle(sel_num_copy)
        train_index = sel_num_copy[:int(0.8 * len(sel_num_copy))]
        test_index = sel_num_copy[int(0.8 * len(sel_num_copy)):]
        
        log(f'Train samples: {len(train_index)}, Test samples: {len(test_index)}')
        log(f'First 5 train indices: {train_index[:5]}')
        log(f'First 5 test indices: {test_index[:5]}')

        # 更新进度条描述
        round_pbar.set_description(f"Round {i+1}/{config.train_test_num}")
        
        # 创建solver并传递日志函数
        solver = WaveHyperSolver(config, folder_path[config.dataset], train_index, test_index, log_func=log)
        srcc_all[i], plcc_all[i] = solver.train()
        
        # 检查是否是全局最优模型
        if srcc_all[i] > global_best_srcc:
            global_best_srcc = srcc_all[i]
            global_best_plcc = plcc_all[i]
            global_best_round = i + 1
            
            # 复制当前最优模型为全局最优模型
            current_best_path = os.path.join(global_save_dir, 'wave_model_best.pth')
            global_best_path = os.path.join(global_save_dir, 'wave_model_global_best.pth')
            
            # 在模型保存部分添加错误处理
            try:
                if os.path.exists(current_best_path):
                    checkpoint = torch.load(current_best_path)
                    torch.save({
                        'model_state_dict': checkpoint['model_state_dict'],
                        'global_best_srcc': global_best_srcc,
                        'global_best_plcc': global_best_plcc,
                        'best_round': global_best_round,
                        'epoch': checkpoint.get('epoch', 0)
                    }, global_best_path)
                    
                    log(f'🎉 New global best model saved! Round {global_best_round}, SRCC: {global_best_srcc:.4f}, PLCC: {global_best_plcc:.4f}')
            except Exception as e:
                log(f'❌ Error saving global best model: {str(e)}')

        round_end = time.time()
        round_time = round_end - round_start
        
        # 计算预估剩余时间
        avg_time_per_round = (round_end - start_all) / (i + 1)
        remaining_rounds = config.train_test_num - (i + 1)
        estimated_remaining_time = avg_time_per_round * remaining_rounds
        
        log(f'Round {i + 1} result: SRCC = {srcc_all[i]:.4f}, PLCC = {plcc_all[i]:.4f}, Time = {round_time:.2f} sec')
        log(f'Current Global Best: SRCC = {global_best_srcc:.4f}, PLCC = {global_best_plcc:.4f} (Round {global_best_round})')
        
        if remaining_rounds > 0:
            log(f'Estimated remaining time: {estimated_remaining_time/60:.1f} minutes ({remaining_rounds} rounds left)')
        
        # 更新进度条后缀信息
        round_pbar.set_postfix({
            'SRCC': f'{srcc_all[i]:.4f}',
            'PLCC': f'{plcc_all[i]:.4f}',
            'Best': f'{global_best_srcc:.4f}',
            'Time': f'{round_time:.1f}s'
        })

    round_pbar.close()
    
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    total_time = time.time() - start_all
    log('=' * 80)
    log('FINAL RESULTS:')
    log(f'Testing median SRCC: {srcc_med:.4f}, median PLCC: {plcc_med:.4f}')
    log(f'Global best SRCC: {global_best_srcc:.4f}, Global best PLCC: {global_best_plcc:.4f} (Round {global_best_round})')
    log(f'Total training time: {total_time / 60:.2f} minutes ({total_time / 3600:.2f} hours)')
    log('=' * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ESPL_LIVE_HDR',
                        help='Support datasets: ESPL_LIVE_HDR')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    main(config)
