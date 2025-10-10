import os
import argparse
import random
import numpy as np
import datetime
import time
from HyerIQASolver import HyperIQASolver

# 设置 CUDA 可见设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置日志目录和文件
log_dir = r'E:\xiazai\hyperIQA-master\log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_log.txt')


# 日志写入函数
def log(msg):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")


def main(config):
    start_all = time.time()  # 记录总开始时间

    # 本地数据集路径
    folder_path = {
        'ESPL_LIVE_HDR': r'E:\data\ESPL_LIVE_HDR_Database\Images',
    }

    img_num = {
        'ESPL_LIVE_HDR': list(range(0, 1811)),
    }

    sel_num = img_num[config.dataset]
    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    log('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))

    for i in range(config.train_test_num):
        log('-' * 60)
        log(f'Round {i + 1} start')
        round_start = time.time()

        random.shuffle(sel_num)
        train_index = sel_num[:int(0.8 * len(sel_num))]
        test_index = sel_num[int(0.8 * len(sel_num)):]

        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

        round_end = time.time()
        log(f'Round {i + 1} result: SRCC = {srcc_all[i]:.4f}, PLCC = {plcc_all[i]:.4f}, Time = {round_end - round_start:.2f} sec')

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    total_time = time.time() - start_all
    log('=' * 60)
    log(f'Testing median SRCC: {srcc_med:.4f}, median PLCC: {plcc_med:.4f}')
    log(f'Total training time: {total_time / 60:.2f} minutes')


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
