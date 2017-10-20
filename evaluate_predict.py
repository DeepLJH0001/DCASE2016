#coding=utf-8
from keras.models import load_model
import sys
from dcnn_train import get_label_dict,get_model,cnn_model,load_dev_traindata
import os
import numpy as np
from PIL import Image
from scipy import stats
import theano
from sklearn.svm import SVC

def load_evaluate_data(cfg_path='../TUT-acoustic-scenes-2016-evaluation/evaluation_setup/'
                        ,data_path='./evaluate/',one_hot=True,Normalization=True):
    if not os.path.exists(data_path):
        print 'please runing feature_extract_demo.py Firstly'
        exit()

    load_file = cfg_path+'evaluate.txt'
    dicts = get_label_dict()
    wav_paths = []
    scene_y = []
    with open(load_file,'rb') as f:
        for line in f:
            ls = line.strip('\r\n').split('	')
            print ls[0],ls[1],dicts.get(ls[1])
            scene_y.append(dicts.get(ls[1]))
            wav_paths.append(data_path + ls[0].split('audio/')[1] + '.jpg')
    scene_x = np.empty((len(scene_y),1,128,2584),dtype='float32')
    scene_y = np.array(scene_y,dtype='uint8')

    for i in range(scene_x.shape[0]):
        scene_x[i,0,:, :] = np.asarray(Image.open(wav_paths[i], 'r').convert('L'), dtype='float32')

    strip = 64
    win_size = 64
    step = int(2584 / strip) - 1

    frame_x = np.empty((scene_x.shape[0] * step, 1, 64, win_size), dtype='float32')
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
    print 'load_evaluate_data ok!'
    print(frame_x.shape[0], 'frame samples')
    print(scene_x.shape[0], 'scene samples')
    return frame_x,scene_x,frame_y,scene_y
def get_evaluate_acc(_model,replace_softmax=False):
    # frame_x, scene_x, frame_y, scene_y = load_evaluate_data()
    # print frame_x.shape,scene_x.shape
    # index = [i for i in range(len(frame_x))]
    # random.shuffle(index)
    # frame_x = frame_x[index]
    # frame_y = frame_y[index]
    #采取EarlyStopping的策略防止过拟合
    # early_stopping = EarlyStopping(monitor='val_acc', patience=5)
    # _model.fit(frame_x,frame_y,batch_size=24,nb_epoch=nb_epoch,callbacks=[early_stopping],validation_split=0.2,shuffle=True)
    # _model.fit(frame_x, frame_y, batch_size=24, nb_epoch=nb_epoch,
    #            shuffle=True)
    frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_evaluate_data()
    frame_acc = _model.evaluate(frame_x_test,frame_y_test)
    print 'evaluate_frame_acc:_'+str(frame_acc)

    lists = []#做混淆矩阵（错分）
    f = open('./11-23改版1_error_evaluation'+ '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        frame_predicts = np.empty((39,),dtype='uint8')
        # for j in range(39):#step of frames in a scene sample
        frame_predicts = _model.predict_classes(frame_x_test[i*39:i*39+39,:,:,:],verbose=0)#np.argmax(_model.predict(frame_x_test[i*39:i*39+39,:,:,:]),axis=-1)
        scene_predict = stats.mode(frame_predicts).mode
        if scene_predict[0] == scene_y_test[i]:
            pass
        else:
            f.write(str(scene_y_test[i])+'错分为->'+str(scene_predict[0])+'\n')
            lists.append([scene_y_test[i],scene_predict[0]])
    f.close()
    scene_acc = float(scene_y_test.shape[0]-len(lists))/float(1.0*scene_y_test.shape[0])
    print 'evaluate_scene_acc_:'+str(scene_acc)
    return frame_acc,scene_acc
def Svc_classify(train_x,train_y,get_cnn_feature):
    print 'SVM traning...'
    svm_clf = SVC(C=1.0,kernel='rbf')
    svm_clf.fit(train_x,train_y)
    frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_evaluate_data(one_hot=False, Normalization=True)
    # frame_x_test = get_cnn_feature(frame_x_test)
    lists = []  # 做混淆矩阵（错分）
    f = open('./error_evaluation_svm' + '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        frame_predicts = np.empty((39,), dtype='uint8')
        # for j in range(39):#step of frames in a scene sample
        frame_predicts = svm_clf.predict(get_cnn_feature([frame_x_test[i * 39:i * 39 + 39, :, :, :],0])[0])
        scene_predict = stats.mode(frame_predicts).mode
        if scene_predict[0] == scene_y_test[i]:
            pass
        else:
            f.write(str(scene_y_test[i]) + '错分为->' + str(scene_predict[0]) + '\n')
            lists.append([scene_y_test[i], scene_predict[0]])
    f.close()
    scene_acc = float(scene_y_test.shape[0] - len(lists)) / float(1.0 * scene_y_test.shape[0])
    print 'evaluate_scene_acc_:' + str(scene_acc)

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('gbk')
    # model = load_model('./model.h5')
    model = get_model()
    model.load_weights('./all_weight.h5')
    print get_evaluate_acc(model)
    # print model.layers
    # import keras.layers
    # from keras import backend as K
    # # get_cnn_feature = theano.function([model.layers[0].input],model.layers[10].output,allow_input_downcast=False)
    # get_cnn_feature = K.function([model.layers[0].input,K.learning_phase()],[model.layers[10].output])
    #
    # # exit()
    # # from keras.models import Model
    # # get_cnn_feature = Model(input=model.input,output=model.layers[10].output)
    # train_x,train_y = load_dev_traindata()
    # train_x = train_x.astype('float32')/255
    # # # frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_evaluate_data(one_hot=False,Normalization=True)
    # print train_x.shape
    # #
    # # train_x =   get_cnn_feature.predict(train_x,batch_size=16)
    # # print train_x.shape
    # train_x = get_cnn_feature([train_x,0])[0]
    # Svc_classify(train_x,train_y,get_cnn_feature)

    # print get_evaluate_acc(model)