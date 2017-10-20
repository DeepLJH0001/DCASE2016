#coding=utf-8
import numpy as np
from PIL import Image
# from keras.preprocessing.image import array_to_img,load_img
import os
import matplotlib.pyplot as plt
#one_hot
from sklearn.feature_extraction import DictVectorizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D


def load_train_data(path = './result/'):
    lists =  [path+list for list in os.listdir(path) if not list.__contains__('c3')]
    print len(lists)
    X_train = np.empty((1170,1,128,2584),dtype='float32')
    y = np.empty((1170,),dtype='float32')
    # img = load_img(i for i in lists,'r')
    n_sample = 0
    for i in lists:
        img = np.asarray(Image.open(i,'r').convert('L'),dtype='float32')
        X_train[n_sample,:,:] = img
        n_sample += 1
        # print img.shape
        # plt.imshow(img)
        # plt.show()
    # X_train = load_img()
    return X_train,y
def get_label_dict():
    dicts = dict()
    dicts['residential_area'] = 0
    #to fix a little bug with string split
    dicts['residential_are'] = 0
    dicts['city_center']=1
    dicts['beach']=2
    dicts['park']=3
    dicts['home'] = 4
    dicts['forest_path'] = 5
    dicts['bus'] = 6
    dicts['grocery_store'] = 7
    dicts['cafe/restaurant'] = 8
    dicts['car'] = 9
    dicts['train'] = 10
    dicts['metro_station'] = 11
    dicts['office'] = 12
    dicts['tram'] = 13
    dicts['library'] = 14
    # print dicts
    return dicts
def load_dev_traindata(cfg_path = '../TUT-acoustic-scenes-2016-development/meta.txt',
                       data_path = './result/',
                       one_hot=True,Normalization=True):
    scene_y = []
    wav_paths = []
    dicts = get_label_dict()
    with open(cfg_path,'rb') as f:
        for line in f:
            ls = line.strip('\n').split('	')
            # print ls
            print ls[0]
            scene_y.append(dicts.get(ls[1]))
            wav_paths.append(data_path+ls[0].split('audio/')[1]+'.jpg')
    f.close()
    print scene_y
    print wav_paths
    # lists = [path + list for list in os.listdir(path) if not list.__contains__('c3')]
    # print len(lists)
    scene_x = np.empty((len(scene_y), 1, 64, 2584), dtype='float32')#2584/8=323   strip=323 win_size = 646
    scene_y = np.array(scene_y,dtype='float32')
    # print y.shape
    from keras.utils import np_utils
    # y = np_utils.to_categorical(y, 15)
    # print y.shape
    for i in range(scene_x.shape[0]):
        scene_x[i,0,:, :] = np.asarray(Image.open(wav_paths[i], 'r').convert('L'), dtype='float32')

    strip = 16
    win_size = 64
    step = int((2584-win_size)/strip)-1

    frame_x = np.empty((scene_x.shape[0] * step, 1, 64, win_size), dtype='float32')
    frame_y = []

    for i in range(scene_x.shape[0]):
        for j in range(step):
            frame_x[i * step + j, :, :, :] = scene_x[i, 0, :, j * strip:j * strip + win_size]
            frame_y.append(scene_y[i])
            # plt.imshow(frame_x[i*step+j,0,:,:],cmap='gray')
            # plt.figure()
            # plt.imshow(scene_x[i,0,:,:],cmap='gray')
            # plt.show()
    frame_y = np.array(frame_y, dtype='uint8')
    if one_hot is True:
        from keras.utils import np_utils
        frame_y = np_utils.to_categorical(frame_y, 15)  # label one-hot
    if Normalization is True:
        frame_x = frame_x.astype('float32') / 255
        scene_x = scene_x.astype('float32') / 255
    print 'load_all_dev_data ok!'
    print(frame_x.shape[0], ' samples')
    return frame_x, scene_x, frame_y, scene_y
def get_model():
    # weights_path = self.weights_path
    from keras.regularizers import l2,l1l2
    model = Sequential()
    # model.add(BatchNormalization(axis=1))
    # model.add(ZeroPadding2D((1, 1), input_shape=(1,128,128)))#layer1
    conv1 = Convolution2D(32, 3, 3,border_mode='same',input_shape=(1,64,64),bias=False)#原来是64
    model.add(conv1)    #layer2
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(ZeroPadding2D((1, 1)))    #layer3
    model.add(MaxPooling2D((2,2)))
    conv2 = Convolution2D(64, 3, 3,border_mode='same',bias=False,W_regularizer=l2(0.001))#原来是64
    model.add(conv2)    #layer4
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3,border_mode='same',bias=False,activation='relu',W_regularizer=l2(0.001)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2))) ##layer5
    # model.add(BatchNormalization(axis=0))
    # model.add(Dropout(0.25))
    '''
    model.add(ZeroPadding2D((1, 1)))    ##layer6
    conv3 = Convolution2D(64, 3, 3,W_regularizer=l1l2(0.002))#原来是128
    model.add(conv3)    ##layer7
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))    #layer8
    conv4 = Convolution2D(64, 3, 3,W_regularizer=l1l2(0.002)) #缘来是128
    model.add(conv4)    #layer9
    # model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2))) #layer10
    # model.add(Dropout(0.25))
    # model.add(BatchNormalization(axis=0))
    '''
    model.add(Flatten(name='cnn_feature_output')) #layer11
    # model.add(BatchNormalization())
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    from keras.optimizers import Adam
    # Lr = 0.001
    adam = Adam()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    return model
def cnn_model():
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(1,128,128),activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def get_localtime():
    import time
    times = time.localtime()
    string_ = str(times.tm_mday) + '_day' + str(times.tm_hour) + '_hour' + str(times.tm_min) + '_min'
    print string_
    return string_
if __name__ == '__main__':
    # X_train, y=load_dev_traindata()

    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    #
    data, label = load_dev_traindata()
    data = data.astype('float32') / 255
    import random

    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]

    from keras.utils import np_utils
    label = np_utils.to_categorical(label, 15)#label one-hot
    print(data.shape[0], ' samples')
    # import pickle

    # kfold = StratifiedKFold(n_splits=4, shuffle=True)  # 4 fold
    # sk_model = KerasClassifier(build_fn=get_model, nb_epoch=30, batch_size=1, verbose=1)
    # score = cross_val_score(sk_model, data, label, cv=kfold)
    # print 'batch_size = 16', score,
    # print 'mean', score.mean()
    model = get_model()
    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor='acc', patience=2)
    model.fit(data,label,nb_epoch=50,batch_size=16,callbacks=[early_stopping])
    time_ = get_localtime()
    model.save_weights('./all_weight.h5')
    # model.save(time_.encode('utf-8')+'d_cnn.h5')
    # get_label_dict()