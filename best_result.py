#coding=utf-8
import numpy as np
from keras.models import load_model,model_from_yaml
import os
import matplotlib.pyplot as plt
from PIL import Image
from dcnn_train import get_model,cnn_model,load_dev_traindata
import random
from scipy import stats
from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.models import load_weights

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
                        ,data_path='./result64/',one_hot=True,Normalization=True):
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
            # print ls[0],ls[1],dicts.get(ls[1])
            scene_y.append(dicts.get(ls[1]))
            wav_paths.append(data_path + ls[0].split('audio/')[1] + '.jpg')
    scene_x = np.empty((len(scene_y),1,64,2584),dtype='float32')
    scene_y = np.array(scene_y,dtype='uint8')

    for i in range(scene_x.shape[0]):
        scene_x[i,0,:, :] = np.asarray(Image.open(wav_paths[i], 'r').convert('L'), dtype='float32')

    #
    #  strip = 64
    # win_size = 128
    # step = int(2584 / 64) - 1
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
    print 'load_'+data+'_data ok!'
    print(frame_x.shape[0], ' samples')
    return frame_x,scene_x,frame_y,scene_y

def get_acc(_model,fold=1):
    # frame_x, scene_x, frame_y, scene_y = load_fold_data(data='Train',fold=fold,Normalization=True)

    frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_fold_data(data='Test', fold=fold,Normalization=True)

    # # print frame_x.shape,scene_x.shape
    # index = [i for i in range(len(frame_x))]
    # random.shuffle(index)
    # frame_x = frame_x[index]
    # frame_y = frame_y[index]
    strip = 64
    step = int(2584 / strip) - 1
    # filepath = "best_12_06_02.nn"+str(1)
    filepath = "best_12_06_02.nn" + str(fold)
    _model.load_weights(filepath)
    frame_acc = _model.evaluate(frame_x_test,frame_y_test,verbose=2)
    print 'fold_'+str(fold)+'_frame_acc:_'+str(frame_acc)

    lists = []#做混淆矩阵（错分）
    f = open('./error+' + str(fold) + '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        # frame_predicts = np.empty((39,),dtype='uint8')
        # for j in range(39):#step of frames in a scene sample
        # frame_predicts = np.argmax(_model.predict(frame_x_test[i*39:i*39+39,:,:,:]),axis=-1)
        frame_predicts = _model.predict_classes(frame_x_test[i*step:i*step+step,:,:,:],verbose=0)
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
def get_all_result():
    result = []
    _model = get_model()
    result.append(get_acc(_model, fold=1))
    _model = get_model()
    result.append(get_acc(_model, fold=2))
    _model = get_model()
    result.append(get_acc(_model, fold=3))
    _model = get_model()
    result.append(get_acc(_model, fold=4))
    print result
    print (float(result[0][1])+float(result[1][1])+float(result[2][1])+float(result[3][1]))/4.0

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
            # print ls[0],ls[1],dicts.get(ls[1])
            scene_y.append(dicts.get(ls[1]))
            wav_paths.append(data_path + ls[0].split('audio/')[1] + '.jpg')
    scene_x = np.empty((len(scene_y),1,64,2584),dtype='float32')
    scene_y = np.array(scene_y,dtype='uint8')

    for i in range(scene_x.shape[0]):
        scene_x[i,0,:, :] = np.asarray(Image.open(wav_paths[i], 'r').convert('L'), dtype='float32')

    strip = 64
    win_size = 64
    step = int((2584-win_size )/ strip) - 1

    frame_x = np.empty((scene_x.shape[0] * step, 1, 64, win_size), dtype='float32')
    frame_y = []
    for  i in range(scene_x.shape[0]):
        for j in range(step):
            frame_x[i*step+j,:,:,:]=scene_x[i,0,:,j*strip:j*strip+win_size]
            # print i*step+j,j*strip+win_size,scene_x[i,0,:,j*strip:j*strip+win_size].shape
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
    frame_acc = _model.evaluate(frame_x_test,frame_y_test,verbose=2)
    print 'evaluate_frame_acc:_'+str(frame_acc)
    strip = 64
    win_size = 64
    step = int((2584 - win_size) / strip) - 1

    lists = []#做混淆矩阵（错分）
    f = open('./12-13_error_evaluation'+ '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        frame_predicts = np.empty((39,),dtype='uint8')
        # for j in range(39):#step of frames in a scene sample
        frame_predicts = _model.predict_classes(frame_x_test[i*step:i*step+step,:,:,:],verbose=0)#np.argmax(_model.predict(frame_x_test[i*39:i*39+39,:,:,:]),axis=-1)
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
if __name__ == '__main__':
    # get_all_result()获取4-fold的精度
    _model = get_model()
    open('2016_12_13_dcnn_model_architecture.yaml', 'w').write( _model.to_yaml())
    _model = model_from_yaml(open('2016_12_13_dcnn_model_architecture.yaml').read())

    from keras.optimizers import Adam
    adam = Adam()
    _model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )
    _model.load_weights('./2016_12_13_full.nn')
    # print  get_evaluate_acc(_model, replace_softmax=False)
    # frame 75.72% scene 83.85%
    #全程prelu fold_1_frame_acc:_[1.5185656813857085, 0.75539811066126861]
    #fold_1_scene_acc_:0.841025641026

    # exit()
    print _model.layers[8].name#输出cnn的特征输出
    print _model.layers[5].name,_model.layers[5].output_shape
    # exit()
    #载入数据
    frame_x, scene_x, frame_y, scene_y = load_dev_traindata(data_path='./result64/', one_hot=False, Normalization=True)
    frame_x_test, scene_x_test, frame_y_test, scene_y_test = load_evaluate_data(one_hot=False)
    #方法一生成cnn特征
    from keras import backend as K
    get_cnn_feature = K.function([_model.layers[0].input, K.learning_phase()], [_model.layers[8].output])

    # dev_cnn_feature=np.empty((frame_x.shape[0],1,16384),dtype='float32')
    dev_dense_featrue = np.empty((frame_x.shape[0],1,512),dtype='float32')#第九层截断
    #截断dense层(9),这样得到的维数比较低 512维度
    get_dense_feature = K.function([_model.layers[0].input, K.learning_phase()], [_model.layers[9].output])
    print _model.layers[9].name
    print get_dense_feature([frame_x[0].reshape(1,1,64,64),0])[0].shape
    # exit()
    for i in range(frame_x.shape[0]):
        # plt.imshow(frame_x[i].reshape(64,64),cmap='gray')# plt.show()
        # open('./cnn_feature','w').write(str(get_cnn_feature([frame_x[i].reshape(1,1,64,64),0])[0]))
        # print get_cnn_feature([frame_x[i].reshape(1,1,64,64),0])[0]
        # print get_cnn_feature([frame_x[i].reshape(1,1,64,64),0])[0].shape
        # dev_cnn_feature[i,:,:]=get_cnn_feature([frame_x[i].reshape(1,1,64,64),0])[0]
        dev_dense_featrue[i,:,:]=get_dense_feature([frame_x[i].reshape(1,1,64,64),0])[0]
    # dev_cnn_feature = dev_cnn_feature.reshape(frame_x.shape[0],16384)
    dev_dense_featrue = dev_dense_featrue.reshape(frame_x.shape[0],512)
    #方法二 提取cnn的中间层特征
    # from keras.models import Model
    # get_cnn_feature = Model(input=_model.input, output=_model.layers[8].output)
    # print get_cnn_feature.predict(frame_x,batch_size=1)
    #方法三 提取cnn中间层特征
    # get_featuremap = theano.function([model.layers[0].input], model.layers[2].output, allow_input_downcast=False)

    # evaluate_cnn_feature = np.empty((frame_y_test.shape[0],1,16384),dtype='float32')
    evaluate_dense_feature = np.empty((frame_y_test.shape[0],1,512),dtype='float32')
    for i in range(frame_x_test.shape[0]):
        # evaluate_cnn_feature[i,:,:] = get_cnn_feature([frame_x_test[i].reshape(1,1,64,64),0])[0]
        evaluate_dense_feature[i,:,:] = get_dense_feature([frame_x_test[i].reshape(1,1,64,64),0])[0]
    # evaluate_cnn_feature = evaluate_cnn_feature.reshape(frame_y_test.shape[0],16384)
    evaluate_dense_feature = evaluate_dense_feature.reshape(frame_y_test.shape[0],512)
    print 'get cnn feature ok!'
    #进行二阶段训练
    '''
    from keras.regularizers import l2
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    from keras.layers.advanced_activations import PReLU
    next_model = Sequential()
    next_model.add(Dense(512,input_shape=(16384L,),W_regularizer=l2(0.002)))
    next_model.add(PReLU())
    next_model.add(Dropout(0.5))
    next_model.add(Dense(512))
    next_model.add(PReLU())
    next_model.add(Dropout(0.5))
    next_model.add(Dense(15,activation='softmax'))
    next_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    #onehot
    from keras.utils import np_utils
    frame_y = np_utils.to_categorical(frame_y,15)
    frame_y_test = np_utils.to_categorical(frame_y_test,15)
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
    filepath = 'best_next_weight2016_12_16.nn'
    _callbacks = [ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=0,save_best_only=True)
                  ,ReduceLROnPlateau(monitor='val_acc',verbose=0,patience=10,factor=0.5)
                  ]
    # next_model.fit(dev_cnn_feature,frame_y,nb_epoch=500,verbose=2,batch_size=512,callbacks=_callbacks,validation_data=(evaluate_cnn_feature,frame_y_test))

    # next_model.load_weights('best_next_weight2016_12_16.nn')
    # print next_model.evaluate(evaluate_cnn_feature,frame_y_test)

    # exit()
    '''
    # 替换LDA为分类器
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # other_classify = LinearDiscriminantAnalysis()
    # other_classify.fit(dev_cnn_feature, frame_y)
    # print 'LDA', other_classify.score(evaluate_cnn_feature, frame_y_test)  # LDA 不能处理大数据
    # other_classify.fit(dev_dense_featrue, frame_y)
    # print 'LDA', other_classify.score(evaluate_dense_feature, frame_y_test)  #LDA 未激活 0.680836707152 激活后：
                                                                            #第二层Dense 未激活 LDA 0.694331983806 激活 739068825911
    # exit()
    #替换svm为分类器
    # from sklearn.svm import SVC
    # other_classify = SVC()
    # other_classify.fit(dev_dense_featrue,frame_y)
    # print 'SVM',other_classify.score(evaluate_dense_feature,frame_y_test)#svm 未激活 0.750607287449
                                                                        #第二层Dense 未激活 0.757962213225 激活  0.755195681511

    # exit()
    #替换RF为分类器
    # from sklearn.ensemble import RandomForestClassifier
    # other_classify = RandomForestClassifier(n_estimators=400,oob_score=True,n_jobs=8,verbose=False)#max_depth=20
    # other_classify.fit(dev_dense_featrue,frame_y)
    # print 'RF',other_classify.score(evaluate_dense_feature,frame_y_test)#RFmax depth=20 未激活 75.877%  激活 0.756275303644
                                                                        #RFmax depth=None 未激活  0.76032388664 激活  0.758502024291
    # svm_classify.                                                     max depth=20 第二层 未激活 0.761538461538 激活  0.761268556005
                                                                        #max depth = None oob_score=True 第二层 未激活  0.76322537112激活 0.761470985155
                                                                        #maxdepth = None oob_score=False 第二层 未激活  0.76322537112 激活 0.762010796221

    #RF最好的是86.41%

    # exit()
    #替换QDA为分类器
    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # other_classify = QuadraticDiscriminantAnalysis()
    # other_classify.fit(dev_dense_featrue,frame_y)
    # print 'QDA',other_classify.score(evaluate_dense_feature, frame_y_test) #QDA 未激活 0.661943319838 激活 0.681916329285
                                                                            #第二层Dense QDA 未激活 0.716261808367

    #替换KNN
    from sklearn.neighbors import KNeighborsClassifier
    other_classify = KNeighborsClassifier()
    other_classify.fit(dev_dense_featrue,frame_y)
    print 'KNN:',other_classify.score(evaluate_dense_feature,frame_y_test)#KNN 未激活 0.70 激活 0.731713900135
                                                                        #第二层  未激活 0.751754385965 激活 0.743994601889
    # exit()
    # 保存模型
    from sklearn.externals import joblib
    # joblib.dump(other_classify,"./model/cnn_QDA_model.m")
    # other_classify = joblib.load("cnnRF_model.m")
    # print frame_y_test.shape,evaluate_cnn_feature.shape
    # print other_classify.score(evaluate_cnn_feature,frame_y_test)

    # exit()
    strip = 64
    win_size = 64
    step = int((2584 - win_size) / strip) - 1

    lists = []  # 做混淆矩阵（错分）
    f = open('./01-10_cnn+svm_confusion_matrix' + '.txt', 'w')  # 写入错分日志
    for i in range(scene_x_test.shape[0]):
        # for j in range(39):#step of frames in a scene sample
        frame_predicts = other_classify.predict(evaluate_dense_feature[i * step:i * step + step, :]) #其他分类器预测 # np.argmax(_model.predict(frame_x_test[i*39:i*39+39,:,:,:]),axis=-1)
        # frame_predicts = next_model.predict_classes(evaluate_cnn_feature[i * step:i * step + step,:])#三阶段DNN预测
        scene_predict = stats.mode(frame_predicts).mode
        if scene_predict[0] == scene_y_test[i]:
            f.write(str(scene_y_test[i]) + '正分为->' + str(scene_predict[0]) + '\n')
        else:
            f.write(str(scene_y_test[i]) + '错分为->' + str(scene_predict[0]) + '\n')
            lists.append([scene_y_test[i], scene_predict[0]])
    f.close()
    scene_acc = float(scene_y_test.shape[0] - len(lists)) / float(1.0 * scene_y_test.shape[0])
    print 'rf_evaluate_scene_acc_:' + str(scene_acc)

    exit()
    print  get_evaluate_acc(_model,replace_softmax=False)
