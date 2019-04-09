
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Average
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from keras import applications
from keras.applications import ResNet50
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
import keras_metrics
from sklearn import metrics
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from tqdm import tqdm
import os, sys
from keras.engine.topology import Input
import h5py
from keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import classification_report, confusion_matrix




from keras import optimizers


# trainLabelsPath = "traininglabels.csv"
# trainImagesPath = "/Users/HS/Downloads/widsdatathon2019/train_images/"
# 
# testFolderPath = "/Users/HS/Downloads/widsdatathon2019/leaderboard_test_data"
# testImagesPath = "/Users/HS/Downloads/widsdatathon2019/leaderboard_test_data/"
# 
# holdFolderPath = "/Users/HS/Downloads/widsdatathon2019/leaderboard_holdout_data"
# holdImagesPath = "/Users/HS/Downloads/widsdatathon2019/leaderboard_holdout_data/"
# 
# subFilePath = "finalSubmission_underSamp_ResNet50_TrainAll.csv"

# In[ ]:



trainLabelsPath = "/home/ubuntu/WIDS/traininglabels.csv"
trainImagesPath = "/home/ubuntu/WIDS/train_images/"

testFolderPath = "/home/ubuntu/WIDS/leaderboard_test_data"
testImagesPath = "/home/ubuntu/WIDS/leaderboard_test_data/"

holdFolderPath = "/home/ubuntu/WIDS/leaderboard_holdout_data"
holdImagesPath = "/home/ubuntu/WIDS/leaderboard_holdout_data/"

subFilePath = "finalSubmission_ResNet50_CV_ConfusionMatrix.csv"


# # Upload training data file

# In[50]:


df = pd.read_csv(trainLabelsPath)


# In[51]:


df.info()


# In[52]:


df[df.has_oilpalm == 1].count() # only 6% of the images have palm oil plantations


# In[53]:


df.score.sort_values()[:5]
# some images have very small scores. Perhaps use a threshold like 0.5 and give 0.5 or less, label 0


# In[54]:


df = df.sort_values(by='image_id').reset_index(drop=True)


# In[55]:


df_hasPalm = df[(df.has_oilpalm == 1)]
df_noPalm = df[(df.has_oilpalm == 0) & (df.score == 1)]


# In[56]:


df_noPalm_samp = df_noPalm.sample(1000, random_state=42)
df_noPalm_samp.shape


# In[57]:


df_final = df_hasPalm.append(df_noPalm_samp).sort_values(by='image_id').reset_index(drop=True)


# In[58]:


df_final.head()


# In[59]:


type(df_final.image_id)


# In[60]:


img_height=256
img_width=256


# In[61]:


# Reading images from a folder and converting to numpy array
train_image = []
for i in tqdm(range(len(df_final))):
    img = image.load_img(trainImagesPath + df_final.image_id[i], target_size=(img_height,img_width))
    img = image.img_to_array(img)
    img = img/255
    
    
    train_image.append(img)
X = np.array(train_image)
X


# In[62]:


y = df_final['has_oilpalm']
#y = to_categorical(y)
y


# # Leaderboard Test Data Prediction

# In[63]:


testpath = testFolderPath
testimagesList = sorted(os.listdir(testpath))
testimagesList


# In[64]:


leaderboard_test_pred = pd.DataFrame(testimagesList , columns=['image_id'])
leaderboard_test_pred


# In[65]:


test_image = []
for i in tqdm(range(len(testimagesList))):
    img = image.load_img(testImagesPath + testimagesList[i], target_size=(img_height,img_width))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
test


# # Leaderboard Holdback Data Prediction

# In[66]:


holdpath = holdFolderPath
holdimagesList = sorted(os.listdir(holdpath))
holdimagesList


# In[67]:


leaderboard_hold_pred = pd.DataFrame(holdimagesList , columns=['image_id'])
leaderboard_hold_pred


# In[68]:


hold_image = []
for i in tqdm(range(len(holdimagesList))):
    img = image.load_img(holdImagesPath + holdimagesList[i], target_size=(img_height,img_width))
    img = image.img_to_array(img)
    img = img/255
    hold_image.append(img)
hold = np.array(hold_image)
hold


# In[69]:


test_final = test_image+hold_image
test_final = np.array(test_final)


# In[70]:


test_set = leaderboard_test_pred.append(leaderboard_hold_pred)


# In[75]:


img_height = 256
img_width = 256
epochs = 25


# def resnet50(model_input):   
#     #ReNet50 model definition
#     base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)
#     last = base_model.output
#     x = BatchNormalization()(last)
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     #x = Dropout(0.5)(x)
# 
# 
#     #Freeze
#     for layer in base_model.layers:
#         layer.trainable = False
# 
#     preds = Dense(1, activation='sigmoid')(x)
#     model = Model(base_model.input, preds)
#     
#     #Compile & fit
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
#                       metrics = [keras_metrics.precision(), keras_metrics.recall(),'accuracy'])
#     
#     model.fit(X, y, epochs=epochs, verbose=1)
#     
#     #Unfreeze & fit
#     for layer in model.layers:
#         layer.trainable = True
#     
#     #model.fit(X, y, epochs=epochs, verbose=1)
#     return model
# 
# def InceptionV3(model_input):
#     base_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,input_tensor=model_input)
#     last = base_model.output
#     x = BatchNormalization()(last)
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024, activation='relu')(x)
#     #x = Dropout(0.5)(x)
#     
#     #Freeze
#     for layer in base_model.layers:
#         layer.trainable = False
#    
#     preds = Dense(1, activation='sigmoid')(x)
#     model = Model(base_model.input, preds)
#     
#     #Compile & fit
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
#                       metrics = [keras_metrics.precision(), keras_metrics.recall(),'accuracy'])
#     
#     model.fit(X, y, epochs=epochs, verbose=1)
#     
#     #Unfreeze & fit
#     for layer in model.layers:
#         layer.trainable = True
#     
#     #model.fit(X, y, epochs=epochs, verbose=1)
#     return model
# 
# model_input = Input(shape=(256, 256, 3))
# resnet50_model = resnet50(model_input)
# InceptionV3_model = InceptionV3(model_input)
# 
# ensembled_models = [resnet50_model,InceptionV3_model]
# def ensemble(models,model_input):
#     outputs = [model.outputs[0] for model in models]
#     y = Average()(outputs)
#     model1 = Model(model_input,y,name='ensemble')
#     return model1
# 
# 
# ensemble_model = ensemble(ensembled_models,model_input)
# #ensemble_model.summary()

# In[ ]:


def resnet50(model_input):   
    #ReNet50 model definition
    base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)
    last = base_model.output
    x = BatchNormalization()(last)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, preds)
    return model

model_input = Input(shape=(256, 256, 3))
resnet_model = resnet50(model_input)


# In[44]:


model = resnet_model


# In[47]:


def train_model(model, batch_size, epochs, img_size, x, y, test, n_fold, kf):
    roc_auc = metrics.roc_auc_score
    preds_train = np.zeros(len(x), dtype = np.float)
    preds_test = np.zeros(len(test), dtype = np.float)
    train_scores = []; valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x):
        #print(train_index, ' ',test_index)
        x_train = x[train_index]; x_valid = x[test_index]
        y_train = y[train_index]; y_valid = y[test_index]


        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 
                               verbose=1, min_lr=1e-7),
             ModelCheckpoint(filepath='inception.fold_' + str(i) + '.hdf5', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = len(test) / batch_size
        
        model = resnet_model

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
                      metrics = [keras_metrics.precision(), keras_metrics.recall(),'accuracy'])

        model.fit(x_train, y_train, epochs=epochs, verbose=1,
                             callbacks=callbacks, batch_size=16,
                             validation_data=(x_valid, y_valid))

        model.load_weights(filepath='inception.fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict(x_valid,
                                       verbose=1)[:, 0]

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict(x_train,
                                       verbose=1)[:, 0]

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Avg Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))

        print('Running train Confusion Matrix with fold {}'.format(i))
        train_y_class = preds_train.argmax(axis=-1)
        
        print("\n Classification Report \n"+ classification_report(y_train, train_y_class))

        print("\n Confusion Matrix \n")
        print(confusion_matrix(y_train,train_y_class))
        
        
        print('Running valid Confusion Matrix with fold {}'.format(i))
        valid_y_class = preds_valid.argmax(axis=-1)
    
        print("\n Classification Report \n"+ classification_report(y_valid, valid_y_class))

        print("\n Confusion Matrix \n")
        print(confusion_matrix(y_valid,valid_y_class))
    
    
        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict(test_final,
                                               verbose=1)[:, -1]

        preds_test += preds_test_fold

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold


    return preds_test


# In[46]:


batch_size = 5
n_fold = 4
img_size = (img_height, img_width)
kf = KFold(n_splits=n_fold, shuffle=True)

test_pred = train_model(model, batch_size, epochs, img_size, X, y, test_final, n_fold, kf)

test_set['has_oilpalm'] = test_pred
test_set.to_csv(subFilePath, index = None)

