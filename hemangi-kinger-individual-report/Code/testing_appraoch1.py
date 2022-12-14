import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# ------------------------------------------------------------------------------------------------------------------
# Environment Variables ****** DO NOT TOUCH *******
# ------------------------------------------------------------------------------------------------------------------

#  0, 1, 3
# Review documentation on tensorflow https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=False)  # Path of file
parser.add_argument("--split", default=False, type=str, required=False)  # validate, test, train

args = parser.parse_args()

PATH = '/home/ubuntu/ML2/Project'  # args.path
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
TRAIN_DIR = DATA_DIR + 'train'
VALID_DIR = DATA_DIR + 'valid'
TEST_DIR = DATA_DIR + 'test'
SPLIT = args.split
os.chdir(PATH + os.path.sep + 'Code' + os.path.sep)

mlb = MultiLabelBinarizer()
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

# ------------------------------------------------------------------------------------------------------------------
# Hyper parameters
# ------------------------------------------------------------------------------------------------------------------

CHANNELS = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
lr = 10 ** -6
n_epoch = 25

# Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

NICKNAME = 'Group2'


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
        xreal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
                j = []
                for x in joined_string:
                    if x != ',':
                        j.append(int(x))
                    else:
                        continue
                xreal.append(np.array(j, dtype='int32'))
            xdf_data['target_class'] = xfinal
            xdf_data['real_label'] = xreal


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


# ------------------------------------------------------------------------------------------------------------------


def read_data(num_classes, phase):
    # Only the training set
    # xdf_dset ( data set )
    # read the data from the file

    if phase == 'train':
        ds_inputs = np.array(TRAIN_DIR + os.path.sep + xdf_dset['id'])
    elif phase == 'valid':
        ds_inputs = np.array(VALID_DIR + os.path.sep + xdf_dset['id'])
    else:
        ds_inputs = np.array(TEST_DIR + os.path.sep + xdf_dset['id'])

    ds_targets = get_target(num_classes)

    # Make the channel as a list to make it variable
    # Create the data set and call the function map to create the dataloader using
    # tf.data.Dataset.map
    # Map creates an iterable
    # More information on https://www.tensorflow.org/tutorials/images/classification

    list_ds = tf.data.Dataset.from_tensor_slices(
        (ds_inputs, ds_targets))  # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds


# ------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):
    """
             feature is the path and id of the image
             target is the result
             returns the image and the target as label
    """

    label = target

    # gamma = 2.0

    # Processing feature

    # load the raw data from the file as a string
    file_path = feature

    # Read the image from disk
    # Make some augmentation if possible

    img = tf.io.read_file(file_path)

    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])

    # augmentation - Reshape the image to get the right dimensions for the initial input in the model

    img = tf.reshape(img, [-1])

    img = img / 255

    return img, label


# ------------------------------------------------------------------------------------------------------------------


def get_target(num_classes):
    """
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    """

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target


# ------------------------------------------------------------------------------------------------------------------

def save_model(model):
    """
         receives the model and print the summary into a .txt file
    """

    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# ------------------------------------------------------------------------------------------------------------------

def model_definition():
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # Define the first dense layer
    model.add(
        tf.keras.layers.Dense(500, activation='LeakyReLU', input_shape=(INPUTS_r,), kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))

    # middle layers
    model.add(tf.keras.layers.Dense(500, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(500, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(500, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(300, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(300, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(200, activation='LeakyReLU'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))

    # final layer , OUTPUTS_a is the number of targets

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    save_model(model)  # print Summary
    return model


# ------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    """
        train the model
    """

    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format(NICKNAME), monitor='accuracy',
                                                     save_best_only=True)

    # EarlyStopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor='accuracy')

    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=50, monitor='accuracy')

    final_model = model_definition()

    # final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    final_model.fit(train_ds, epochs=n_epoch, validation_data=valid_ds,
                    callbacks=[check_point, early_stopping_cb, reduce_lr_on_plateau_cb])


# ------------------------------------------------------------------------------------------------------------------


def predict_func(test_ds):
    """
        predict function
    """

    final_model = tf.keras.models.load_model('model_{}.h5'.format(NICKNAME))
    res = final_model.predict(test_ds)

    save_model(final_model)

    xres = [tf.argmax(f).numpy() for f in res]
    res[res >= THRESHOLD] = 1
    res[res < THRESHOLD] = 0
    res = res.astype('int32')
    resp = [res[i] for i in range(len(res))]
    # resp = [list(x) for x in resp]
    xdf_dset['res_pred'] = resp
    xdf_dset['results'] = xres
    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    final_model.evaluate(test_ds)

    return res

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

# Main
if __name__ == "__main__":

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

    # Multilabel , verify the classes , change from strings to numbers

    class_names = process_target(2)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    # Processing Train dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    train_ds = read_data(OUTPUTS_a, 'train')

    xdf_dset = xdf_data[xdf_data["split"] == 'valid'].copy()

    valid_ds = read_data(OUTPUTS_a, 'valid')

    train_func(train_ds)

    # Preprocessing Test dataset

    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds = read_data(OUTPUTS_a, 'test')
    y_pred = predict_func(test_ds)

    # Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_weighted', 'coh']
    list_of_agg = ['avg']

    # metrics_func(list_of_metrics, list_of_agg, xdf_dset['real_label'].values, y_pred)
