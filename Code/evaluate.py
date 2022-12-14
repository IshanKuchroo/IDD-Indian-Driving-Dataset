# ------------------------------------------------------------------------------------------------------------------
import time

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from metric import metrics_func
from train import model_definition

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
