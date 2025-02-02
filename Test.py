import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader,Dataset
from DataLoader import *
from Metrics import *
from NAC_Loss import *
from Model.LiteMamba_Bound import litemamba_bound 

test_dataset = ISICLoader(type='test', data_path=DATA_PATH, transform=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = DiceLoss(device=self.device)(y_pred, y_true)
        dice_score = dice(y_true, y_pred)
        iou_score = jaccard(y_true, y_pred)
        precision_score = precision(y_true, y_pred)
        recall_score = recall(y_true, y_pred)

        metrics = {
            "Test Dice": dice_score,
            "Test IoU": iou_score,
            "Test Precision": precision_score,
            "Test Recall": recall_score,
            "Test Loss": loss
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

model = litemamba_bound()
model.eval()

#Prediction
DATA_PATH = ''
CHECKPOINT_PATH = ''
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)

#Visualization with Boundary lineline

def visualize_prediction_VT(dataset, idx):
    plt.figure(figsize=(20, 8), layout='compressed')
    x1, y1 = dataset[idx[0]]
    x2, y2 = dataset[idx[1]]
    x3, y3 = dataset[idx[2]]
    x4, y4 = dataset[idx[3]]
    x5, y5 = dataset[idx[4]]
    x6, y6 = dataset[idx[5]]
    x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0), x5.unsqueeze(0), x6.unsqueeze(0))).cuda()
    y = torch.cat((y1.unsqueeze(0), y2.unsqueeze(0), y3.unsqueeze(0), y4.unsqueeze(0), y5.unsqueeze(0), y6.unsqueeze(0))).cuda()

    y_pred1 = model1(x).data.squeeze()
    y_pred2 = model2(x).data.squeeze()
    y_pred3 = model3(x).data.squeeze()
    y_pred4 = model4(x).data.squeeze()
    y_pred5 = model5(x).data.squeeze()
    y_pred6 = model6(x).data.squeeze()

    y_pred1 = torch.argmax(y_pred1, dim=1).cpu().numpy().astype(np.uint8)
    y_pred2 = torch.argmax(y_pred2, dim=1).cpu().numpy().astype(np.uint8)
    y_pred3 = torch.argmax(y_pred3, dim=1).cpu().numpy().astype(np.uint8)
    y_pred4 = torch.argmax(y_pred4, dim=1).cpu().numpy().astype(np.uint8)
    y_pred5 = torch.argmax(y_pred5, dim=1).cpu().numpy().astype(np.uint8)
    y_pred6 = torch.argmax(y_pred6, dim=1).cpu().numpy().astype(np.uint8)

    predictions = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
    titles = []

    for i in range(6):
        # Convert input tensor to numpy array for visualization
        xa = x[i].permute(1, 2, 0).cpu().numpy()
        ya = (y[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).squeeze()  # Ensure it's single-channel
        gt_contours, _ = cv2.findContours(ya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Display the input image with contours for each model
        for j, (pred, title) in enumerate(zip(predictions, titles)):
            contour_img = xa.copy()  # Copy the input image
            contours, _ = cv2.findContours(pred[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # Green contour
            cv2.drawContours(contour_img, gt_contours, -1, (50, 0, 255), 2)  # Light red for GT, thickness = 1
            plt.subplot(6, 8, 8 * i + j + 2)
            if i == 0:
                plt.title(title)
            plt.imshow(contour_img)
            plt.axis('off')

        # Display the original input image (without contour) in the first column
        plt.subplot(6, 8, 8 * i + 1)
        if i == 0:
            plt.title("Image")
        plt.imshow(xa)
        plt.axis('off')

        # Display ground truth in the last column
        ya = y[i].permute(1, 2, 0).cpu().numpy()
        plt.subplot(6, 8, 8 * i + 8)
        if i == 0:
            plt.title("Ground Truth")
        plt.imshow(ya, cmap='gray')
        plt.axis('off')
        
#Visual
device = 'cuda'
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)
model4 = model4.to(device)
model5 = model5.to(device)
model6 = model6.to(device)
visualize_prediction_VT(test_dataset,idx = [])
