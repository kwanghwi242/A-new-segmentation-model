import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from DataLoader import *
from Metrics import *
from NAC_Loss import *
from Model.LiteMamba_Bound import litemamba_bound 


# Lightning module
import timm.optim
class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = NAC_LossS(self.device)(y_true, y_pred)
        dice_score, jaccard_score = dice(y_true, y_pred), jaccard(y_true, y_pred)
        return loss, dice_score, jaccard_score

    def training_step(self, batch, batch_idx):
        loss, dice_score, jaccard_score = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice_score, "train_jac": jaccard_score}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_score, jaccard_score = self._step(batch)
        metrics = {"val_loss": loss, "val_dice": dice_score, "val_jac":jaccard_score}
        self.log_dict(metrics, prog_bar = True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice_score, jaccard_score = self._step(batch)
        metrics = {"test_loss": loss, "test_dice": dice_score, "test_jac":jaccard_score}
        self.log_dict(metrics, prog_bar = True)
        return metrics

    def configure_optimizers(self):
        optimizer = timm.optim.NAdam(self.parameters(), lr=1e-3)
        #optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=8, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers
    
model = litemamba_bound().cuda()

DATA_PATH = ''

# Dataset & Data Loader
train_dataset = ISICLoader(type='train', data_path=DATA_PATH, transform=True)
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

val_dataset = ISICLoader(type='test', data_path=DATA_PATH, transform=False)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

# Training config
os.makedirs('/content/weights', exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint('/content/weights', filename="ckpt{val_dice:0.4f}_wo_all",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False)
progress_bar = pl.callbacks.TQDMProgressBar()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":True,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":100,
          "precision":16,
          }
trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)

trainer.fit(segmentor, train_loader, val_loader)