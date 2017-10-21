#coding=utf-8


import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,classes2,
                          normalize=False,
                          title='Confusion matrix\n\n\n\n\n',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60,fontsize=15,horizontalalignment='right',verticalalignment='top')
    plt.yticks(tick_marks, classes,fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",#矩阵文本位置
                 verticalalignment="center",
                 # withdash=True,
                 fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout(rect=(0,0,0.75,1))#这个加入就歪了rect = (left, bottom, right, top)
    plt.tight_layout()
    # plt.ylabel('True label',fontsize=17)
    # plt.xlabel('Predicted label',fontsize=17)
def get_label_dict():
    dicts = dict()
    dicts[u'住宅区'] = 0
    dicts[u'城市中心']=1
    dicts[u'湖边的沙滩']=2
    dicts[u'公园']=3
    dicts[u'家庭'] = 4
    dicts[u'森林小径'] = 5
    dicts[u'公共汽车'] = 6
    dicts[u'超市'] = 7
    dicts[u'咖啡/餐馆'] = 8
    dicts[u'普通汽车'] = 9
    dicts[u'火车'] = 10
    dicts[u'地铁站'] = 11
    dicts[u'办公室'] = 12
    dicts[u'有轨电车'] = 13
    dicts[u'图书馆'] = 14
    # print dicts
    return dicts
y_test = []
y_pred = []
# class_names = get_label_dict().keys()
class_names = [u'森林小径',u'办公室',u'超市',u'咖/餐馆',
               u'公共汽车',u'地铁站',u'公园',u'图书馆',u'有轨电车',
               u'住宅区',u'火车',u'普通汽车',u'城市中心',u'家庭',u'沙滩']#\u68ee\u6797\u5c0f\u5f84
class_names2 = []
for i in range(len(class_names)):
    class_names2.append(class_names[i][::-1])
print class_names2
# exit()

with open('12-19_cnn+rf_confusion_matrix.txt','r') as f:
    for i in f:
        y_test.append(i[:-1].split('->')[0])
        y_pred.append(i[:-1].split('->')[1])
# exit()
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix')
plot_confusion_matrix(cnf_matrix, classes=class_names,classes2=class_names2)
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
plt.savefig('./result.jpg')
plt.show()
