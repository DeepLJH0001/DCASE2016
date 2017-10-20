#coding=utf-8
import librosa
import wavio
from scipy import signal
import numpy as np
import os
import matplotlib.pyplot as plt
#list嵌套for循环和if语句
# print [list for list in os.listdir('./') if list.endswith('.wav')]

def read_wav(file):
    data = wavio.read(file)
    x = data.data.astype(float)/np.power(2,data.sampwidth*8-1)
    fs = data.rate
    return x,fs
#提取mel特征
#save_path保存声谱图的地方
#load_dev_path 存放dcase2016声音文件的文件路径
def mel_feature_extract_dev(save_path,load_dev_path,nmel = 128):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lists = [list for list in os.listdir(load_dev_path) if list.endswith('.wav')]
    for i in lists:
        x, fs = read_wav(load_dev_path+i)
        # print fs, x.shape
        if x.ndim == 2:
            x = np.mean(x, axis=-1)#对双声道取平均值
        assert fs == 44100
        #这边是qiuqiangkong的写法，完全可以按照librosa提取声音特征
        # print fs, x.shape
        # ham_win = np.hamming(1024)
        # [f,t,X] = signal.spectrogram(x,window=ham_win,nperseg=1024,noverlap=0,detrend=False,return_onesided=True)
        # print X.shape
        # X = X.T
        # melW = librosa.filters.mel(fs,n_fft=1024,n_mels=64,fmin=0,fmax=22100)
        # melW /=np.max(melW,axis=-1)[:,None]
        # X = np.dot(X,melW.T)
        # print X.shape
        S = librosa.feature.melspectrogram(x, sr=fs,n_mels=nmel)
        # print S.shape
        Log_s = librosa.logamplitude(S, ref_power=np.max)
        # print Log_s.shape
        #验证结果
        # plt.matshow(Log_s,origin='lower',aspect = 'auto')
        # plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        # plt.show()
        Log_s = np.flipud(Log_s)
        # print os.listdir(os.path.curdir)
        # print save_path
        # plt.imshow(Log_s,cmap='gray')
        # plt.imsave(save_path + str(i) + '.jpg', Log_s, cmap='gray')
        # plt.show()
        # exit()
        plt.imsave(save_path + str(i)+'.jpg', Log_s, cmap='gray')
        plt.imsave(save_path + str(i)+'_c3.jpg', Log_s)
def mel_feature_extract_evaluate(save_path,load_evaluate_path,nmel = 128):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lists = [list for list in os.listdir(load_evaluate_path) if list.endswith('.wav')]
    for i in lists:
        x, fs = read_wav(load_evaluate_path+i)
        if x.ndim == 2:
            x = np.mean(x, axis=-1)#对双声道取平均值
        assert fs == 44100
        S = librosa.feature.melspectrogram(x, sr=fs,n_mels=nmel)
        Log_s = librosa.logamplitude(S, ref_power=np.max)
        Log_s = np.flipud(Log_s)
        plt.imsave(save_path + str(i)+'.jpg', Log_s, cmap='gray')
        plt.imsave(save_path + str(i)+'_c3.jpg', Log_s)
if __name__ == '__main__':
   # mel_feature_extract_dev('./result64/','../TUT-acoustic-scenes-2016-development/audio/',nmel=64)
    mel_feature_extract_evaluate('./evaluate/','../TUT-acoustic-scenes-2016-evaluation/audio/',nmel=64)
   # print [list for list in os.listdir('../TUT-acoustic-scenes-2016-development/audio/')]