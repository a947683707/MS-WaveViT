import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
import time
import os  # 添加缺失的导入
from models import TargetNet
from wave_models.wavehypernet import WaveHyperNet
from data_loader import MyDataLoader

class WaveHyperSolver(object):
    """Solver for training and testing WaveHyperNet"""
    def __init__(self, config, path, train_idx, test_idx, log_func=None):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.log_func = log_func if log_func else print

        self.model_hyper = WaveHyperNet().cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model_hyper.backbone.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras = [
            {'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
            {'params': self.model_hyper.backbone.parameters(), 'lr': self.lr},
        ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = MyDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num,
                                    batch_size=config.batch_size, istrain=True)
        test_loader = MyDataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                   istrain=False)

        self.train_loader = train_loader.get_data()
        self.test_loader = test_loader.get_data()
        
        # 添加保存目录配置
        self.save_dir = r"H:\ZhangR\hyperIQA-master\wave_models\saved_models"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 计算总的batch数量用于进度条
        self.total_batches = len(self.train_loader)

    def train(self):
        best_srcc = 0.0
        best_plcc = 0.0
        
        self.log_func('  Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTime(s)\tStatus')
        self.log_func('  ' + '-' * 75)

        # 使用tqdm显示Epoch进度
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs", unit="epoch", leave=False)
        
        for t in epoch_pbar:
            epoch_start = time.time()
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            # 使用tqdm显示batch进度
            batch_pbar = tqdm(enumerate(self.train_loader), 
                            total=self.total_batches, 
                            desc=f"Epoch {t+1}/{self.epochs}", 
                            unit="batch", 
                            leave=False)

            for batch_idx, (img, label) in batch_pbar:
                img = img.cuda()
                label = label.cuda()

                self.solver.zero_grad()
                paras = self.model_hyper(img)

                model_target = TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                pred = model_target(paras['target_in_vec'])
                pred_scores.extend(pred.cpu().tolist())
                gt_scores.extend(label.cpu().tolist())

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                
                # 更新batch进度条
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{np.mean(epoch_loss):.4f}'
                })

            batch_pbar.close()
            
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_srcc, test_plcc = self.test(self.test_loader)
            
            epoch_time = time.time() - epoch_start
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            
            # 判断是否为最优模型
            status = ""
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                status = "💾 BEST"
                
                # 保存当前轮次的最优模型
                save_path = os.path.join(self.save_dir, 'wave_model_best.pth')
                torch.save({
                    'model_state_dict': self.model_hyper.state_dict(),
                    'best_srcc': best_srcc,
                    'best_plcc': best_plcc,
                    'epoch': t + 1
                }, save_path)
                
                self.log_func(f"  💾 Saved best model: SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f}", print_console=False)

            # 记录详细的epoch信息到日志文件
            epoch_info = f"  {t+1:2d}\t{avg_loss:8.4f}\t{train_srcc:8.4f}\t{test_srcc:8.4f}\t{test_plcc:8.4f}\t{epoch_time:6.1f}\t{status}"
            self.log_func(epoch_info)
            
            # 控制台显示简化信息
            print(f'  {t+1:2d}\t{avg_loss:4.3f}\t\t{train_srcc:4.4f}\t\t{test_srcc:4.4f}\t\t{test_plcc:4.4f}\t{status}')

            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'SRCC': f'{test_srcc:.4f}',
                'PLCC': f'{test_plcc:.4f}',
                'Best': f'{best_srcc:.4f}',
                'Loss': f'{avg_loss:.4f}'
            })

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            paras = [
                {'params': self.hypernet_params, 'lr': lr * self.lrratio},
                {'params': self.model_hyper.backbone.parameters(), 'lr': self.lr},
            ]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        epoch_pbar.close()
        
        self.log_func('  ' + '-' * 75)
        self.log_func(f'  Round Best: SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f}')
        return best_srcc, best_plcc

    def test(self, data):
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            img = img.cuda()
            label = label.cuda()

            paras = self.model_hyper(img)
            model_target = TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            pred_scores.append(float(pred.item()))
            gt_scores.extend(label.cpu().tolist())

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc
