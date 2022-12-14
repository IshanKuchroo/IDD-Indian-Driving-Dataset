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
from timm.optim import Lookahead
from torch.optim import RAdam
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import timm
from torch_optimizer import Ranger
import os
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------
'''
LAST UPDATED 11/10/2021, lsdr
02/14/2022 am ldr checking for consistency
02/14/2022 pm ldr class model same as train_solution pytorch, change of numpy() to cpu().numpy()
'''
# ------------------------------------------------------------------------------------------------------------------

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
TRAIN_DIR = DATA_DIR + 'train'
VALID_DIR = DATA_DIR + 'valid'
TEST_DIR = DATA_DIR + 'test'
sep = os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

n_epoch =  40
BATCH_SIZE = 32
LR = 0.0001
DROPOUT = 0.3
CHANNELS = 3
IMAGE_SIZE = 480

# --------------------------------Do Not Change------------------------------------------------------------------------

NICKNAME = "Group2"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

torch.manual_seed(42)
np.random.seed(42)


# ------------------------------------------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------------------------------------------
class Dataset(data.Dataset):
    """
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_IDs, type_data, target_type):
        # Initialization'
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        # Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        elif self.type_data == 'valid':
            y = xdf_dset_valid.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)

            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = TRAIN_DIR + os.path.sep + xdf_dset.id.get(ID)
        elif self.type_data == 'valid':
            file = VALID_DIR + os.path.sep + xdf_dset_valid.id.get(ID)
        else:
            file = TEST_DIR + os.path.sep + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=45, interpolation=transforms.InterpolationMode.NEAREST, fill=0),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # Augmentation only for train

        X = torch.FloatTensor(transform(img))

        # X = torch.FloatTensor(img)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        X /= 255.0

        return X, y


# ------------------------------------------------------------------------------------------------------------------

def read_data(target_type):
    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = xdf_dset['target_class']

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)
    list_of_ids_valid = list(xdf_dset_valid.index)

    # Datasets
    partition = {
        'train': list_of_ids,
        'test': list_of_ids_test,
        'valid': list_of_ids_valid
    }

    # Data Loaders

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True}

    training_set = Dataset(partition['train'], 'train', target_type)
    training_generator = data.DataLoader(training_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    valid_set = Dataset(partition['valid'], 'valid', target_type)
    valid_generator = data.DataLoader(valid_set, **params)

    # Make the channel as a list to make it variable

    return training_generator, test_generator, valid_generator


# ------------------------------------------------------------------------------------------------------------------


def save_model(model):
    """
      Print Model Summary
    """

    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))


# ------------------------------------------------------------------------------------------------------------------


def model_definition(pretrained=False):
    """
        Define a Keras sequential model
        Compile the model
    """
    # Explanability is important - LIME or SHAP or DeepLift or Integrated Gradient or Layered Gradient or Class
    # Activation Map
    if pretrained:
        # model = models.regnet_x_800mf(weights='DEFAULT')
        # model = models.regnet_y_800mf(weights='DEFAULT')
        model = models.regnet_y_400mf(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    else:
        model = CNN()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # base_optim = RAdam(model.parameters(), lr=LR)
    # optimizer = Lookahead(base_optim, k=5, alpha=0.5)

    # optimizer = Ranger(model.parameters(), lr=LR)

    criterion = nn.BCEWithLogitsLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    save_model(model)

    return model, optimizer, criterion, scheduler


# ------------------------------------------------------------------------------------------------------------------


# def train_and_test(train_ds, valid_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
#     print("!!START TRAINING!!")
#
#     model, optimizer, criterion, scheduler = model_definition(pretrained)
#
#     cont = 0
#     train_loss_item = list([])
#     val_loss_item = list([])
#
#     pred_labels_per_hist = list([])
#
#     model.phase = 0
#
#     met_val_best = 0
#
#     # Measure the elapsed time of each epoch
#     t0_epoch, t0_batch = time.time(), time.time()
#
#     for epoch in range(n_epoch):
#
#         train_loss, steps_train = 0, 0
#
#         model.train()
#
#         pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         train_hist = list([])
#         val_hist = list([])
#
#         with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#             for xdata, xtarget in train_ds:
#
#                 xdata, xtarget = xdata.to(device), xtarget.to(device)
#                 output = model(xdata)
#
#                 loss = criterion(output, xtarget)
#                 train_loss += loss.item()
#                 cont += 1
#
#                 steps_train += 1
#
#                 train_loss_item.append([epoch, loss.item()])
#
#                 pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                 if len(pred_labels_per_hist) == 0:
#                     pred_labels_per_hist = pred_labels_per
#                 else:
#                     pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                 if len(train_hist) == 0:
#                     train_hist = xtarget.cpu().numpy()
#                 else:
#                     train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])
#
#                 pbar.update(1)
#                 pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / BATCH_SIZE))
#
#                 # pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss))
#
#                 pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
#                 real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         pred_labels = pred_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         # Metric Evaluation
#         train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         avg_train_loss = train_loss / steps_train
#
#         # Validation Dataset - Loss calculation
#
#         model.eval()
#
#         val_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#         val_loss, steps_val = 0, 0
#         met_val = 0
#
#         with torch.no_grad():
#
#             with tqdm(total=len(valid_ds), desc="Epoch {}".format(epoch)) as pbar:
#
#                 for xdata, xtarget in valid_ds:
#
#                     xdata, xtarget = xdata.to(device), xtarget.to(device)
#                     output = model(xdata)
#
#                     loss = criterion(output, xtarget)
#                     val_loss += loss.item()
#                     cont += 1
#
#                     steps_val += 1
#
#                     val_loss_item.append([epoch, loss.item()])
#
#                     pred_labels_per = output.detach().to(torch.device('cpu')).numpy()
#
#                     if len(pred_labels_per_hist) == 0:
#                         pred_labels_per_hist = pred_labels_per
#                     else:
#                         pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])
#
#                     if len(val_hist) == 0:
#                         val_hist = xtarget.cpu().numpy()
#                     else:
#                         val_hist = np.vstack([val_hist, xtarget.cpu().numpy()])
#
#                     pbar.update(1)
#                     pbar.set_postfix_str("Validation Loss: {:.5f}".format(val_loss / steps_val))
#
#                     # pbar.set_postfix_str("Validation Loss: {:.5f}".format(val_loss))
#
#                     val_logits = np.vstack((val_logits, output.detach().cpu().numpy()))
#                     real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#                     optimizer.zero_grad()
#
#         pred_labels = val_logits[1:]
#         pred_labels[pred_labels >= THRESHOLD] = 1
#         pred_labels[pred_labels < THRESHOLD] = 0
#
#         val_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)
#
#         avg_val_loss = val_loss / steps_val
#
#         # Finish with Training
#
#         print("\n")
#         xstrres = "Epoch {}: ".format(epoch)
#         for met, dat in train_metrics.items():
#             xstrres = xstrres + ' Train ' + met + ' {:.5f}'.format(dat)
#         print("\n")
#         xstrres = xstrres + " - "
#         for met, dat in val_metrics.items():
#             xstrres = xstrres + ' Validation ' + met + ' {:.5f}'.format(dat)
#             if met == save_on:
#                 met_val = dat
#
#         print(xstrres)  # Print metrics
#
#         if met_val > met_val_best and SAVE_MODEL:
#
#             torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
#             xdf_dset_results = xdf_dset_valid.copy()
#
#             # The following code creates a string to be saved as 1,2,3,3,
#             # This code will be used to validate the model
#             xfinal_pred_labels = []
#             for i in range(len(pred_labels)):
#                 joined_string = ",".join(str(int(e)) for e in pred_labels[i])
#                 xfinal_pred_labels.append(joined_string)
#
#             xdf_dset_results['results'] = xfinal_pred_labels
#
#             xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#             print("The model has been saved!")
#             met_val_best = met_val
#
#     # Print performance over the entire training data
#     time_elapsed = time.time() - t0_epoch
#     print("-" * 61)
#     print(f"{'AVG TRAIN LOSS':^12} | {'AVG VAL LOSS':^10} | {'VAL ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
#     print("-" * 61)
#     print(f"{avg_train_loss:^14.6f} | {avg_val_loss:^10.6f} | {met_val_best:^17.2f} | {time_elapsed:^9.2f}")
#     print("-" * 61)
#     print("\n")
#
#     print("!!TRAINING COMPLETE!!")
#

def train_and_test(train_ds, valid_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
    print("!!START TRAINING!!")

    model, optimizer, criterion, scheduler = model_definition(pretrained)

    cont = 0
    train_loss_item = list([])
    val_loss_item = list([])

    pred_labels_per_hist = list([])

    training_stats = []
    train_acc = []
    train_hlm = []
    val_acc = []
    val_hlm = []

    model.phase = 0

    met_val_best = 0

    # Measure the elapsed time of each epoch
    t0_epoch, t0_batch = time.time(), time.time()

    for epoch in range(n_epoch):

        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        train_hist = list([])
        val_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata, xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)
                output = model(xdata)

                loss = criterion(output, xtarget)
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / BATCH_SIZE))

                # pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        # Validation Dataset - Loss calculation

        model.eval()

        val_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

        val_loss, steps_val = 0, 0
        met_val = 0

        with torch.no_grad():

            with tqdm(total=len(valid_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata, xtarget in valid_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    output = model(xdata)

                    loss = criterion(output, xtarget)
                    val_loss += loss.item()
                    cont += 1

                    steps_val += 1

                    val_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(val_hist) == 0:
                        val_hist = xtarget.cpu().numpy()
                    else:
                        val_hist = np.vstack([val_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Validation Loss: {:.5f}".format(val_loss / steps_val))

                    # pbar.set_postfix_str("Validation Loss: {:.5f}".format(val_loss))

                    val_logits = np.vstack((val_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

                    optimizer.zero_grad()

        pred_labels = val_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        val_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_val_loss = val_loss / steps_val

        # Finish with Training

        print("\n")
        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres + ' Train ' + met + ' {:.5f}'.format(dat)
            if met == 'acc':
                train_acc.append(dat)
            elif met == 'hlm':
                train_hlm.append(dat)
        print("\n")
        xstrres = xstrres + " - "
        for met, dat in val_metrics.items():
            xstrres = xstrres + ' Validation ' + met + ' {:.5f}'.format(dat)

            if met == 'acc':
                val_acc.append(dat)
            elif met == 'hlm':
                val_hlm.append(dat)

            if met == save_on:
                met_val = dat

        print(xstrres)  # Print metrics

        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Train Accuracy': train_acc[epoch],
                'Valid Accuracy': val_acc[epoch],
                'Train HLM': train_hlm[epoch],
                'Valid HLM': val_hlm[epoch]
            }
        )

        if met_val > met_val_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))
            xdf_dset_results = xdf_dset_valid.copy()

            # The following code creates a string to be saved as 1,2,3,3,
            # This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
            print("The model has been saved!")
            met_val_best = met_val

    lst = [x for x in range(1, n_epoch)]
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 8)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], marker='o', color='blue', label="Training")
    plt.plot(df_stats['Valid. Loss'], marker='o', color='red', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(lst)

    plt.show()

    # Plot the learning curve.
    plt.plot(df_stats['Train Accuracy'], marker='*', color='blue', label="Training")
    plt.plot(df_stats['Valid Accuracy'], marker='*', color='red', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(lst)

    plt.show()

    # Plot the learning curve.
    plt.plot(df_stats['Train HLM'], marker='*', color='blue', label="Training")
    plt.plot(df_stats['Valid HLM'], marker='*', color='red', label="Validation")

    # Label the plot.
    plt.title("Training & Validation HLM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(lst)

    plt.show()

    # Print performance over the entire training data
    time_elapsed = time.time() - t0_epoch
    print("-" * 61)
    print(f"{'AVG TRAIN LOSS':^12} | {'AVG VAL LOSS':^10} | {'VAL ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
    print("-" * 61)
    print(f"{avg_train_loss:^14.6f} | {avg_val_loss:^10.6f} | {met_val_best:^17.2f} | {time_elapsed:^9.2f}")
    print("-" * 61)
    print("\n")

    print("!!TRAINING COMPLETE!!")

# ------------------------------------------------------------------------------------------------------------------


def metrics_func(metrics, aggregates, y_true, y_pred):
    """
    multiple functions of metrics to call each function
    f1, cohen, accuracy, matthews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    """

    def f1_score_metric(y_true, y_pred, type):
        """
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        """
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum / xcont
    # Ask for arguments for each metric

    return res_dict


# ------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    """
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Multiclass or Multilabel ( binary  ( 0,1 ) )
    :return:
    """

    if target_type == 1:
        # takes the classes and then
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names = (xtarget)
        xdf_data['target_class'] = final_target

    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 3:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_

    # We add the column to the main dataset

    return class_names


# ------------------------------------------------------------------------------------------------------------------

def classes_to_dataframe(xdf_data, mode):
    df = pd.DataFrame()

    df['id'] = xdf_data['filename']

    str_col_lst = []
    tgt_col_lst = []
    for i in range(len(xdf_data)):
        str_col = ''
        lst_col = ''

        if xdf_data[' animal'][i] == 1:
            str_col += 'animal,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' autorickshaw'][i] == 1:
            str_col += 'autorickshaw,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' bicycle'][i] == 1:
            str_col += 'bicycle,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' bus'][i] == 1:
            str_col += 'bus,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' car'][i] == 1:
            str_col += 'car,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' caravan'][i] == 1:
            str_col += 'caravan,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' motorcycle'][i] == 1:
            str_col += 'motorcycle,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' person'][i] == 1:
            str_col += 'person,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' rider'][i] == 1:
            str_col += 'rider,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' traffic light'][i] == 1:
            str_col += 'traffic light,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' traffic sign'][i] == 1:
            str_col += 'traffic sign,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' trailer'][i] == 1:
            str_col += 'trailer,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' train'][i] == 1:
            str_col += 'train,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' truck'][i] == 1:
            str_col += 'truck,'
            lst_col += '1,'
        else:
            lst_col += '0,'

        if xdf_data[' vehicle fallback'][i] == 1:
            str_col += 'vehicle fallback'
            lst_col += '1'
        else:
            lst_col += '0'

        if str_col[-1:] == ',':
            str_col_lst.append(str_col[:len(str_col) - 1])
            tgt_col_lst.append(lst_col[:len(lst_col) - 1])
        else:
            str_col_lst.append(str_col)
            tgt_col_lst.append(lst_col)

    df['target'] = str_col_lst

    if mode == 'train':
        df['split'] = 'train'
    elif mode == 'valid':
        df['split'] = 'valid'
    else:
        df['split'] = 'test'

    # df['target_class'] = tgt_col_lst

    return df


def evaluate_model(test_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
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


def shap_explainablity(data):

    batch = next(iter(data))
    images, _ = batch

    background = images[:10]
    test_images = images[10:15]

    model, optimizer, criterion, scheduler = model_definition(pretrained=True)
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    background, test_images = background.to(device), test_images.to(device)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.detach().cpu().numpy(), 1, -1), 1, 2)

    shap.image_plot(shap_numpy, -test_numpy * 255)


# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    modes = ['train', 'valid', 'test']

    for mode in modes:
        # Reading the Excel file from a directory

        if mode == 'train':
            for file in os.listdir(TRAIN_DIR):
                if file[-4:] == '.csv':
                    FILE_NAME = TRAIN_DIR + os.path.sep + file

            # Reading and filtering Excel file
            xdf_data = pd.read_csv(FILE_NAME)

            df_train = classes_to_dataframe(xdf_data, mode)

        elif mode == 'valid':
            for file in os.listdir(VALID_DIR):
                if file[-4:] == '.csv':
                    FILE_NAME = VALID_DIR + os.path.sep + file

            # Reading and filtering Excel file
            xdf_data = pd.read_csv(FILE_NAME)

            df_valid = classes_to_dataframe(xdf_data, mode)

        else:
            for file in os.listdir(TEST_DIR):
                if file[-4:] == '.csv':
                    FILE_NAME = TEST_DIR + os.path.sep + file

            # Reading and filtering Excel file
            xdf_data = pd.read_csv(FILE_NAME)

            df_test = classes_to_dataframe(xdf_data, mode)

    xdf_data = df_train.append(df_valid, ignore_index=True)

    xdf_data = xdf_data.append(df_test, ignore_index=True)

    # Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    class_names = process_target(target_type=2)

    # Comment

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    xdf_dset_valid = xdf_data[xdf_data["split"] == 'valid'].copy()

    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    # read_data creates the dataloaders, take target_type = 2

    train_ds, test_ds, valid_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['acc', 'hlm']
    list_of_agg = ['sum']

    # train_and_test(train_ds, valid_ds, list_of_metrics, list_of_agg, save_on='acc', pretrained=True)

    evaluate_model(test_ds, list_of_metrics, list_of_agg, save_on='acc', pretrained=True)

    # shap_explainablity(test_ds)

    batch = next(iter(test_ds))
    images, _ = batch

    background = images[:5]
    test_images = images[:1]

    model, optimizer, criterion, scheduler = model_definition(pretrained=True)
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    background, test_images = background.to(device), test_images.to(device)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.detach().cpu().numpy(), 1, -1), 1, 2)

    shap.image_plot(shap_numpy, -test_numpy * 255)

    shap.image_plot(shap_numpy, -test_numpy )

    # shap.image_plot(shap_numpy* 255, -test_numpy)
