import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from utils.dataset_utils import PromptTrainDataset
from dataloader import HW4TrainDataset, HW4TestDataset
from net.model_new import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
from utils.image_io import save_image_tensor
# from utils.val_utils import AverageMeter, compute_psnr_ssim
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=150
        )

        return [optimizer], [scheduler]


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    ckpt_path = "./train_ckpt/epoch=119-step=96000.ckpt"

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    test_dataset = HW4TestDataset("../hw4_realse_dataset/test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # psnr = AverageMeter()
    # ssim = AverageMeter()

    with torch.no_grad():
        for (name, degrad_patch) in tqdm(test_loader):
            print(name)
            degrad_patch = degrad_patch.cuda()

            restored = net(degrad_patch)

            save_image_tensor(restored, './output/' + name[0])
        # print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
