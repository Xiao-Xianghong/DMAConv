import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.optim.lr_scheduler import StepLR
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from DMAConv_UNet import MANet
import numpy as np
import h5py
import shutil
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 使用无图形界面后端，避免服务器报错
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    print("num of visible CUDA:", torch.cuda.device_count())
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, using CPU.")

# ================== Pre-test =================== #
def load_set(file_path):
    with h5py.File(file_path, 'r') as f:
    # tensor type:
        lms = np.array(f['lms']).transpose(2, 1, 0) / 2047.0  # 256x256x8
        ms = np.array(f['ms']).transpose(2, 1, 0) / 2047.0  # 64x64x8
        pan = np.array(f['pan']).transpose(1, 0) / 2047.0  # 256x256

    # 转为 PyTorch 张量，并保持原有维度变换
    lms = torch.from_numpy(lms).permute(2, 0, 1).contiguous()  # 8x256x256
    ms = torch.from_numpy(ms).permute(2, 0, 1).contiguous()  # 8x64x64
    pan = torch.from_numpy(pan).contiguous()  # 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    with h5py.File(file_path, 'r') as f:
        # 查看有哪些变量（可选）
        # print(list(f.keys()))

        # 读取 gt 并转置为 HxWxC（如果需要）
        gt = np.array(f['gt']).transpose(2, 1, 0) / 2047.0  # HxWxC = 256x256x8

    # 转为 tensor，并转为 CxHxW
    test_gt = torch.from_numpy(gt).permute(2, 0, 1).contiguous()  # 8x256x256

    return test_gt
# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

## ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0006
#lr = 0.0006 * 0.9
epochs = 600
#epochs = 500
ckpt = 50
#ckpt = 10
batch_size = 32
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MANet().to(device)
'''
pretrain_path = "/root/autodl-tmp/weights/MAConv_UNet_model22_epoch50.pth"
if os.path.isfile(pretrain_path):
    model.load_state_dict(torch.load(pretrain_path, map_location=device))
    print(f"Loaded pretrained weights: {pretrain_path}")
'''
'''
def sam_loss(sr, gt, eps=1e-8):
    """
    输入: sr, gt  [B, C, H, W]
    返回: 平均光谱角 (弧度)
    """
    sr_norm = F.normalize(sr, dim=1, eps=eps)
    gt_norm = F.normalize(gt, dim=1, eps=eps)
    cos_sim = (sr_norm * gt_norm).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_sim).mean()

def ergas_loss(sr, gt, ratio=4):
    """
    输入: sr, gt  [B, C, H, W]
    ratio: PAN/MS 分辨率比，WV-3 为 4
    返回: ERGAS (无单位)
    """
    rmse = torch.sqrt(((sr - gt) ** 2).mean(dim=(2, 3)))        # [B, C]
    mean_gt = gt.mean(dim=(2, 3)) + 1e-6                        # [B, C]
    rel_rmse = (rmse / mean_gt) ** 2
    return 100. / ratio * torch.sqrt(rel_rmse.mean())

mse_fun = nn.MSELoss()

def criterion(sr, gt):
    loss_mse = mse_fun(sr, gt).to(device)
    #loss_sam = sam_loss(sr, gt).to(device)
    #loss_ergas = ergas_loss(sr, gt).to(device)
    return loss_mse #+ 0.1 * loss_sam + 0.1 * loss_ergas
'''
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1
'''
warmup_epochs = 8
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs   # linear warmup
    else:
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
'''
'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=0.0006
)
'''
scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
    shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
writer = SummaryWriter('train_logs')

def save_checkpoint(model, epoch):  # save model function
    save_dir = '/Data2/xianghong_xiao/weights/'
    model_out_path = os.path.join(save_dir, f'MAConv_UNet_model15_fix_epoch{epoch}.pth')
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved: {model_out_path}")

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader):
    print('Start training...')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, _, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                                     Variable(batch[1]).to(device), \
                                     batch[2], \
                                     batch[3], \
                                     Variable(batch[4]).to(device)
            optimizer.zero_grad()  # fixed

            out = model(pan, lms)

            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch


            loss.backward()  # fixed
            optimizer.step()  # fixed

        #lr_scheduler.step()  # update lr
        scheduler.step()

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        train_loss_list.append(t_loss)

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        # if epoch % 10 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         output1, output2, output3 = model(test_ms, test_pan,test_lms)
        #         result_our = torch.squeeze(output3).permute(1, 2, 0)
        #         #sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
        #         result_our = result_our * 2047
        #         result_our = result_our.type(torch.DoubleTensor).to(device)
        #
        #         our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
        #         print('our_SAM: {} dmdnet_SAM: 2.9355'.format(our_SAM) ) # print loss for each epoch
        #         print('our_ERGAS: {} dmdnet_ERGAS:1.8119 '.format(our_ERGAS))
        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, _, _ ,pan= Variable(batch[0], requires_grad=False).to(device), \
                                         Variable(batch[1]).to(device), \
                                         batch[2], \
                                         batch[3], \
                                         Variable(batch[4]).to(device)


                out = model(pan, lms)


                loss = criterion(out, gt) 
                epoch_val_loss.append(loss.item())



        v_loss = np.nanmean(np.array(epoch_val_loss))
        writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))
        val_loss_list.append(v_loss)

    #------------保存loss曲线图-------------#
    loss_dir = "/Data2/xianghong_xiao/train_logs/"
    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PanSharpening Loss Curve MAConv_UNet_model15_fix(All epochs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(loss_dir, 'pansharpening_loss_curve_maconv_unet_model15_fix.png'))
    plt.close()
    
    plt.figure()
    plt.plot(range(51, len(train_loss_list) + 1), train_loss_list[50:], label='Train Loss')
    plt.plot(range(51, len(val_loss_list) + 1), val_loss_list[50:], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PanSharpening Loss Curve MAConv_UNet_model15_fix(After 50 Epochs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(loss_dir, 'pansharpening_loss_curve_maconv_unet_model15_fix_after50.png'))
    plt.close()
    
    # ------------保存loss日志-------------#
    np.savetxt(os.path.join(loss_dir, 'train_loss_maconv_unet_model15_fix.txt'), np.array(train_loss_list))
    np.savetxt(os.path.join(loss_dir, 'val_loss_maconv_unet_model15_fix.txt'), np.array(val_loss_list))
    print("Train/Valid Loss 已保存为 ./train_logs/train_loss_maconv_unet.txt 和 ./train_logs/val_loss_maconv_unet.txt")
    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    train_set = Dataset_Pro("/Data2/xianghong_xiao/training_data/train_wv3.h5")  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro("/Data2/xianghong_xiao/training_data/valid_wv3.h5")  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    # ------------------- load_test ----------------------------------#

    #file_path = "/Data2/XianghongXiao/TestHxWxC_qb_data14.mat"
    #test_lms, test_ms, test_pan = load_set(file_path)
    #test_lms = test_lms.to(device).unsqueeze(dim=0).float()
    #test_ms = test_ms.to(device).unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    #test_pan = test_pan.to(device).unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
    #test_gt= load_gt_compared(file_path)  ##compared_result
    #test_gt = (test_gt * 2047).to(device).double()

    ###################################################################
    #train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
