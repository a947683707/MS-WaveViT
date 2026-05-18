import torch
from scipy import stats
import numpy as np
from tqdm import tqdm
import time
import os
from models import TargetNet
from wave_models.wavehypernet import WaveHyperNet, EnhancedWaveHyperNet_Stage2
from data_loader import MyDataLoader

class WaveHyperSolver(object):
    """Solver for training and testing WaveHyperNet"""
    def __init__(self, config, path, train_idx, test_idx, log_func=None):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.log_func = log_func if log_func else print
        
        # 使用增强版模型（第二阶段）
        self.model_hyper = EnhancedWaveHyperNet_Stage2().cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        # 获取不同类型的参数
        backbone_params = list(map(id, self.model_hyper.backbone.parameters()))
        new_module_params = []
        for module in self.model_hyper.get_new_modules():
            new_module_params.extend(list(map(id, module.parameters())))
        
        # 分离新增模块参数和其他超网络参数
        all_param_ids = backbone_params + new_module_params
        self.hypernet_params = filter(lambda p: id(p) not in all_param_ids, self.model_hyper.parameters())
        self.new_module_params = filter(lambda p: id(p) in new_module_params, self.model_hyper.parameters())

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        
        # 渐进式训练标志
        self.progressive_training = True
        self.stage1_epochs = max(3, self.epochs // 4)  # 第一阶段占1/4的训练周期
        self.stage2_epochs = self.epochs - self.stage1_epochs

        # 初始化优化器（第一阶段：只训练新增模块）
        self.setup_stage1_optimizer()

        train_loader = MyDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num,
                                    batch_size=config.batch_size, istrain=True)
        test_loader = MyDataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                   istrain=False)

        self.train_loader = train_loader.get_data()
        self.test_loader = test_loader.get_data()
        
        self.dataset_name = config.dataset
        self.save_dir = r"C:\Users\PC\Desktop\fsdownload\archive\hyperIQA-master\wave_models\saved_models"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.total_batches = len(self.train_loader)
    
    def setup_stage1_optimizer(self):
        """设置第一阶段优化器：只训练新增的多尺度小波注意力模块"""
        self.model_hyper.freeze_pretrained_modules()
        
        # 只优化新增模块
        new_params = []
        for module in self.model_hyper.get_new_modules():
            new_params.extend(list(module.parameters()))
        
        paras = [{'params': new_params, 'lr': self.lr * self.lrratio}]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        
        self.log_func(f"🔧 第一阶段：只训练新增的多尺度小波注意力模块 ({len(new_params)} 个参数)")
    
    def setup_stage2_optimizer(self):
        """设置第二阶段优化器：端到端训练整个网络"""
        self.model_hyper.unfreeze_all_modules()
        
        backbone_params = list(map(id, self.model_hyper.backbone.parameters()))
        hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        
        paras = [
            {'params': hypernet_params, 'lr': self.lr * self.lrratio},
            {'params': self.model_hyper.backbone.parameters(), 'lr': self.lr},
        ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        
        self.log_func(f"🔧 第二阶段：端到端训练整个网络")

    def train(self):
        best_srcc = 0.0
        best_plcc = 0.0
        
        self.log_func('  阶段\tEpoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTime(s)\tStatus')
        self.log_func('  ' + '-' * 85)

        # 第一阶段：只训练新增模块
        self.log_func(f"\n🚀 开始第一阶段训练 (Epochs 1-{self.stage1_epochs}): 只训练多尺度小波注意力模块")
        stage1_pbar = tqdm(range(self.stage1_epochs), desc="Stage 1", unit="epoch", leave=False)
        
        for t in stage1_pbar:
            epoch_start = time.time()
            epoch_loss, train_srcc = self.train_epoch(t + 1, "Stage1")
            test_srcc, test_plcc = self.test(self.test_loader)
            
            epoch_time = time.time() - epoch_start
            
            status = ""
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                status = "💾 BEST"
                self.save_model(best_srcc, best_plcc, t + 1, "stage1")
            
            epoch_info = f"  S1\t{t+1:2d}\t{epoch_loss:8.4f}\t{train_srcc:8.4f}\t{test_srcc:8.4f}\t{test_plcc:8.4f}\t{epoch_time:6.1f}\t{status}"
            self.log_func(epoch_info)
            
            stage1_pbar.set_postfix({
                'SRCC': f'{test_srcc:.4f}',
                'PLCC': f'{test_plcc:.4f}',
                'Best': f'{best_srcc:.4f}',
                'Loss': f'{epoch_loss:.4f}'
            })
        
        stage1_pbar.close()
        
        # 切换到第二阶段
        self.log_func(f"\n🔄 切换到第二阶段训练 (Epochs {self.stage1_epochs+1}-{self.epochs}): 端到端微调")
        self.setup_stage2_optimizer()
        
        # 第二阶段：端到端训练
        stage2_pbar = tqdm(range(self.stage2_epochs), desc="Stage 2", unit="epoch", leave=False)
        
        for t in stage2_pbar:
            epoch_start = time.time()
            epoch_loss, train_srcc = self.train_epoch(self.stage1_epochs + t + 1, "Stage2")
            test_srcc, test_plcc = self.test(self.test_loader)
            
            epoch_time = time.time() - epoch_start
            
            status = ""
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                status = "💾 BEST"
                self.save_model(best_srcc, best_plcc, self.stage1_epochs + t + 1, "stage2")
            
            epoch_info = f"  S2\t{self.stage1_epochs + t+1:2d}\t{epoch_loss:8.4f}\t{train_srcc:8.4f}\t{test_srcc:8.4f}\t{test_plcc:8.4f}\t{epoch_time:6.1f}\t{status}"
            self.log_func(epoch_info)
            
            stage2_pbar.set_postfix({
                'SRCC': f'{test_srcc:.4f}',
                'PLCC': f'{test_plcc:.4f}',
                'Best': f'{best_srcc:.4f}',
                'Loss': f'{epoch_loss:.4f}'
            })
            
            # 第二阶段的学习率调整
            if (self.stage1_epochs + t + 1) > 8:
                lr = self.lr / pow(10, ((self.stage1_epochs + t + 1) // 6))
                lrratio = 1 if (self.stage1_epochs + t + 1) > 8 else self.lrratio
                
                backbone_params = list(map(id, self.model_hyper.backbone.parameters()))
                hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
                
                paras = [
                    {'params': hypernet_params, 'lr': lr * lrratio},
                    {'params': self.model_hyper.backbone.parameters(), 'lr': self.lr},
                ]
                self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        
        stage2_pbar.close()
        
        self.log_func('  ' + '-' * 85)
        self.log_func(f'  🎯 渐进式训练完成: Best SRCC={best_srcc:.4f}, Best PLCC={best_plcc:.4f}')
        return best_srcc, best_plcc
    
    def train_epoch(self, epoch_num, stage):
        """训练一个epoch"""
        epoch_loss = []
        pred_scores = []
        gt_scores = []

        batch_pbar = tqdm(enumerate(self.train_loader), 
                        total=self.total_batches, 
                        desc=f"{stage} Epoch {epoch_num}", 
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
            pred_scores.extend(pred.cpu().flatten().tolist())
            gt_scores.extend(label.cpu().tolist())

            loss = self.l1_loss(pred.squeeze(), label.float().detach())
            epoch_loss.append(loss.item())
            loss.backward()
            self.solver.step()
            
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{np.mean(epoch_loss):.4f}'
            })

        batch_pbar.close()
        
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        
        return avg_loss, train_srcc
    
    def save_model(self, best_srcc, best_plcc, epoch, stage):
        """保存模型"""
        save_path = os.path.join(self.save_dir, f'wave_model_best_{self.dataset_name}.pth')
        torch.save({
            'model_state_dict': self.model_hyper.state_dict(),
            'best_srcc': best_srcc,
            'best_plcc': best_plcc,
            'epoch': epoch,
            'stage': stage,
            'dataset': self.dataset_name
        }, save_path)
        
        self.log_func(f"  💾 Saved best model for {self.dataset_name}: SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f} ({stage})", print_console=False)

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
