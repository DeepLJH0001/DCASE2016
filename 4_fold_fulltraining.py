#coding=utf-8
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt
from PIL import Image
from dcnn_train import get_model,cnn_model,load_dev_traindata
import random
from scipy import stats
from keras.callbacks import EarlyStopping,ModelCheckpoint
from best_result import load_evaluate_data

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

def load_fold_data(data='Train',fold=1,cfg_path='../TUT-acoustic-scenes-2016-development/evaluation_setup/'
                        ,data_path='./result/',one_hot=True,Normalization=True):
    if not os.path.exists(data_path):
        print 'please runing feature_extract_demo.py Firstly'
        exit()
    if data=='Test':
        load_file = cfg_path+'fold'+str(fold)+'_evaluate.txt'
    else:
        load_file = cfg_path+'fold'+str(fold)+'_train.txt'
    dicts = get_label_dict()
    wav_paths = []
    scene_y = []
    with open(load_file,'rb') as f:
        for line in f:
            ls = line.strip('\r\n').split('	')
            print ls[0],ls[1],dicts.get(ls[1])
            scene_y.append(dicts.get(ls[1]))
            wav_paths.append(data_path + ls[0].split('audio/')[1] + '.jpg')
    scene_x = np.empty((len(scene_y),1,64,2584),dtype='float32')
    scene_y = np.array(scene_y,dtype='uint8')

    for i in range(scene_x.shape[0]):
        scene_x[i,0,:, :] = np.asarray(Image.open(wav_paths[i], 'r').convert('L'), dtype='float32')

    strip = 16
    win_size = 64
    step = int((2584 - win_size) / strip) - 1

    frame_x = np.empty((scene_x.shape[0] * step, 1, 128, win_size), dtype='float32')
    frame_y = []
    for  i in range(scene_x.shape[0]):
        for j in range(step):
            frame_x[i*step+j,:,:,:]=scene_x[i,0,:,j*strip:j*strip+win_size]
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
        scene_x = scene_x.astype('float32') /255
    print 'load_'+data+'_data ok!'
    print(frame_x.shape[0], ' samples')
    return frame_x,scene_x,frame_y,scene_y


def full_train(_model,fold=1,nb_epoch=500,is_EarlyStopping = True):
    # frame_x, scene_x, frame_y, scene_y = load_fold_data(data='Train',fold=fold)
    frame_x, scene_x, frame_y, scene_y = load_dev_traindata(data_path='./result64/')
    frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_evaluate_data()
    # print frame_x.shape,scene_x.shape
    # index = [i for i in range(len(frame_x))]
    # random.shuffle(index)
    # frame_x = frame_x[index]
    # frame_y = frame_y[index]
    filepath = '2016_12_18_full.nn'
    #采取EarlyStopping的策略防止过拟合
    if is_EarlyStopping is True:
        # early_stopping = EarlyStopping(monitor='acc', patience=0)
        early_stopping = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=0, save_best_only=True)
        _model.fit(frame_x,frame_y,verbose=2,batch_size=64,nb_epoch=nb_epoch,callbacks=[early_stopping],shuffle=True,validation_data=(frame_x_test,frame_y_test))
    else:
        _model.fit(frame_x, frame_y, batch_size=16, nb_epoch=nb_epoch,
               shuffle=True,verbose=2)

    _model.load_weights(filepath)
    frame_acc = _model.evaluate(frame_x_test,frame_y_test)
    print 'fold_'+str(fold)+'_frame_acc:_'+str(frame_acc)

    strip = 64
    win_size = 64
    step = int((2584 - win_size) / strip) - 1
    lists = []#做混淆矩阵（错分）
    f = open('./error+' + str(fold) + '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        frame_predicts = np.empty((39,),dtype='uint8')
        # for j in range(39):#step of frames in a scene sample
        # frame_predicts = np.argmax(_model.predict(frame_x_test[i*step:i*step+step,:,:,:]),axis=-1)
        frame_predicts = _model.predict_classes(frame_x_test[i * step:i * step + step, :, :, :], verbose=0)
        scene_predict = stats.mode(frame_predicts).mode
        if scene_predict[0] == scene_y_test[i]:
            pass
        else:
            f.write(str(scene_y_test[i])+'错分为->'+str(scene_predict[0])+'\n')
            lists.append([scene_y_test[i],scene_predict[0]])
    f.close()
    scene_acc = float(scene_y_test.shape[0]-len(lists))/float(1.0*scene_y_test.shape[0])
    print 'fold_'+str(fold)+'_scene_acc_:'+str(scene_acc)
    return frame_acc,scene_acc
if __name__ == '__main__':
    # model = get_model()

    from keras.regularizers import l2, l1l2
    from keras.layers.advanced_activations import PReLU
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

    model = Sequential()
    conv1 = Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 64, 64), bias=False)  # 原来是64
    model.add(conv1)  # layer2
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D((2, 2)))
    conv2 = Convolution2D(64, 3, 3, border_mode='same', bias=False, W_regularizer=l2(0.001))  # 原来是64
    model.add(conv2)  # layer4
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3, border_mode='same', bias=False, W_regularizer=l2(0.001)))
    model.add(PReLU())
    model.add(MaxPooling2D((2, 2)))  ##layer5
    model.add(Flatten(name='cnn_feature_output'))  # layer11
    # model.add(BatchNormalization())
    model.add(Dense(512,W_regularizer=l2(0.002)))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    # model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    from keras.optimizers import Adam

    # Lr = 0.001

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    full_train(_model=model)
    # load_all_dev_data()
    # import sys
    # reload(sys)
    # print sys.getdefaultencoding()
    # sys.setdefaultencoding('gbk')#????????why?
    # name = ['./'+i for i in os.listdir('./') if i.endswith('h5')]

    #下面是载入数据的预处理
    # frame_x, scene_x, frame_y, scene_y=load_fold_data(fold=1)
    # frame_x = frame_x.astype('float32') / 255
    # scene_x = scene_x.astype('float32') /255
    # from keras.utils import np_utils
    # frame_y = np_utils.to_categorical(frame_y, 15)  # label one-hot
    # print(frame_x.shape[0], ' samples')

    #test sth
    # frame_x_test, scene_x_test, frame_y_test, scene_y_test, wav_paths = load_fold_data(data='Test', fold=1)
    # print frame_x_test.shape,wav_paths[1]
    # plt.imshow(frame_x_test[39,0,:,:],cmap='gray')
    # plt.figure()
    # plt.imshow(scene_x_test[1,0,:,:],cmap='gray')
    # plt.show()
    # print wav_paths[1]
    # exit()

    # acc = []
    # model = get_model()
    # model.load_weights('./4_fold_1_model.h5')
    # acc.append(_4_fold_train(model, fold=1))
    # model.save_weights('./4_fold_1_full_model.h5')
    # model = get_model()
    # model.load_weights('./4_fold_2_model.h5')
    # acc.append(_4_fold_train(model, fold=2))
    # model.save_weights('./4_fold_2_full_model.h5')
    # model = get_model()
    # model.load_weights('./4_fold_3_model.h5')
    # acc.append(_4_fold_train(model, fold=3))
    # model.save_weights('./4_fold_3_full_model.h5')
    # model = get_model()
    # model.load_weights('./4_fold_4_model.h5')
    # acc.append(_4_fold_train(model, fold=4))
    # model.save_weights('./4_fold_4_full_model.h5')
    # print acc #4_fold_acc batch=16  n_epoch=20 or 10?
    #[[1.8550737950485663, 0.68125552608311224], [0.72153158176286947, 0.84217506631299732],
    #[0.3474898756968216, 0.93477886766477369], [0.22463585421383897, 0.94766420793818051]]
    #mean = 0.8514 这个是fram-based

    #18_day15_hour2_mindcnn.h5报错 这个是由于保存的时候是gbk保存
    #载入之前的non-full的训练权重
    # model = load_model('./model.h5')
    # acc = model.evaluate(frame_x,frame_y)
    # print model.metrics_names
    # print acc
    #predicts显示的是frame-based的问题
