# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef, \
    classification_report
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os
import seaborn as sns
from matplotlib import pyplot as plt

from metric import metrics_func
# from train import model_definition

n_epoch = 40
BATCH_SIZE = 32
LR = 0.0001
DROPOUT = 0.3
CHANNELS = 3
IMAGE_SIZE = 224

# --------------------------------Do Not Change------------------------------------------------------------------------

NICKNAME = "Group2"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

torch.manual_seed(42)
np.random.seed(42)

OUTPUTS_a = 16


# ------------------------------------------------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=2)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 127, 127)

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 62, 62)

        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3))
        self.convnorm4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.convnorm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d((2, 2))

        self.conv6 = nn.Conv2d(256, 512, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # output (n_examples, 256, 58, 58)

        self.linear1 = nn.Linear(512, 400)  # input will be flattened to (n_examples, 256)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)

        self.linear2 = nn.Linear(400, OUTPUTS_a)
        self.act1 = torch.relu
        self.act2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.act1(self.conv3(x))))
        x = self.pool4(self.convnorm4(self.act1(self.conv4(x))))
        x = self.pool5(self.convnorm5(self.act1(self.conv5(x))))
        x = self.act1(self.conv6(x))
        x = self.global_avg_pool(x).view(-1, 512)
        x = self.drop(self.linear1_bn(self.act1(self.linear1(x))))
        return self.act2(self.linear2(x))


def model_definition(pretrained=False):
    """
        Define a Keras sequential model
        Compile the model
    """
    # Explanability is important - LIME or SHAP or DeepLift or Integrated Gradient or Layered Gradient or Class Activation Map
    if pretrained:
        model = models.resnet101(weights='DEFAULT')
        # model = models.vgg19(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
        # model.classifier[6] = nn.Linear(model.classifier[6].in_features, OUTPUTS_a)
    else:
        model = CNN()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = nn.BCEWithLogitsLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # save_model(model)

    return model, optimizer, criterion, scheduler


def evaluate_model(test_ds, xdf_dset_test, OUTPUTS_a, list_of_metrics, list_of_agg, save_on, pretrained=False):

    print("!!START PREDICTION!!")

    classes = ['unknown', 'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'caravan', 'motorcycle', 'person', 'rider',
               'traffic light', 'traffic sign', 'trailer', 'train', 'truck', 'vehicle fallback']
    model, optimizer, criterion, scheduler = model_definition(pretrained)

    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    model.phase = 0

    n_t_epoch = 1

    cont = 0
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0

    t0_epoch, t0_batch = time.time(), time.time()

    for epoch in range(n_t_epoch):

        # Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        test_loss, steps_test = 0, 0
        met_test = 0
        test_hist = list([])

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata, xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    output = model(xdata)

                    loss = criterion(output, xtarget)
                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        test_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    # pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                    optimizer.zero_grad()

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        print("\n")
        xstrres = "Epoch {}: ".format(epoch)

        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)  # Print metrics

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_test.copy()

            # The following code creates a string to be saved as 1,2,3,3,
            # This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
            print("The model has been saved!")
            met_test_best = met_test

    # Print performance over the entire training data
    time_elapsed = time.time() - t0_epoch
    print("-" * 61)
    print(f"{'AVG TEST LOSS':^10} | {'Test ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
    print("-" * 61)
    print(f"{avg_test_loss:^10.6f} | {met_test_best:^17.2f} | {time_elapsed:^9.2f}")
    print("-" * 61)
    print("\n")

    print('Classification Report for IDD Data :\n',
          classification_report(real_labels[1:], pred_labels, target_names=classes))

    print("!!PREDICTION COMPLETE!!")

    return met_test_best