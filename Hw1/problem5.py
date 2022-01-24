from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import sys
from matplotlib.pyplot import figure

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(932, 797)
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(160, 50, 581, 641))
        self.textEdit.setObjectName("textEdit")

        self.one = QtWidgets.QPushButton(Form)
        self.one.setGeometry(QtCore.QRect(250, 90, 361, 81))
        self.one.setObjectName("one")
        self.one.clicked.connect(one_clicked) #1

        self.two = QtWidgets.QPushButton(Form)
        self.two.setGeometry(QtCore.QRect(250, 190, 361, 81))
        self.two.setObjectName("two")
        self.two.clicked.connect(two_clicked) #2

        self.three = QtWidgets.QPushButton(Form)
        self.three.setGeometry(QtCore.QRect(250, 300, 361, 81))
        self.three.setObjectName("three")
        self.three.clicked.connect(three_clicked) #3

        self.four = QtWidgets.QPushButton(Form)
        self.four.setGeometry(QtCore.QRect(250, 400, 361, 81))
        self.four.setObjectName("four")
        self.four.clicked.connect(four_clicked) #4


        self.five = QtWidgets.QPushButton(Form)
        self.five.setGeometry(QtCore.QRect(250, 590, 361, 81))
        self.five.setObjectName("five")
        

        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(260, 520, 351, 20))
        self.lineEdit.setObjectName("lineEdit")
        

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.textEdit.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt;\">VGG16 TEST</span></p></body></html>"))
        self.one.setText(_translate("Form", "1. SHOW TRAIN IMAGES"))
        self.two.setText(_translate("Form", "2. SHOW HYPERPARAMETER"))
        self.three.setText(_translate("Form", "3. SHOW MODEL SHORTCUT"))
        self.four.setText(_translate("Form", "4. SHOW ACCURACY"))
        self.five.setText(_translate("Form", "5. TEST"))

def one_clicked():
    def readphoto(file):
        import pickle as cPickle
        with open(file,'rb')as fo:
            dict=cPickle.load(fo,encoding='bytes')
        return dict
    cifar10_data=readphoto(r"C:\Users\Windows\Desktop\Hw1\cifar-10-batches-py\data_batch_1") 
    image=cifar10_data[b'data'] 
    labelname=cifar10_data[b'labels']
    photo=int(np.shape(image)[0])
    for i in range (1,10):
        x=randrange(photo)
        pic=image[x].reshape(3,32,32)
        pic=pic.transpose(1,2,0)
        theirname=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        plt.subplot(3,3,i)
        plt.title(theirname[labelname[x]])
        plt.imshow(pic)
    plt.show()
    
def two_clicked():
    print('')
    print('hyperparameters:')
    print('batch size:32')
    print('learning rate: 0.001')
    print('optimizer: SGD')

def three_clicked():
    import torch.nn as nn
    import torchvision.models as models
    from torchvision import models
    model = models.vgg16(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(4096,10)
    print(model)

def four_clicked():
    figure(figsize=(8,4))
    loss_ = [0.43997, 0.22333, 0.14297, 0.09161, 0.05977,
            0.04264, 0.02994, 0.02416, 0.02065, 0.01653,
            0.01289, 0.01120, 0.00910, 0.00744, 0.00584,
            0.00628, 0.00449, 0.00329, 0.00324, 0.00252,
            0.00235, 0.00507, 0.00292, 0.00360, 0.00119]
    acc_train = [84.869, 92.420, 95.082, 96.821, 97.923,
                 98.564, 98.940, 99.202, 99.290, 99.432,
                 99.562, 99.602, 99.696, 99.758, 99.806,
                 99.796, 99.854, 99.900, 99.900, 99.912,
                 99.918, 99.840, 99.912, 99.874, 99.968]
    acc_valid = [89.517, 92.372, 92.053, 93.311, 92.931,
                 93.450, 93.421, 93.540, 93.161, 93.810,
                 94.079, 93.580, 94.369, 93.830, 94.249,
                 94.149, 93.950, 94.159, 94.139, 94.479,
                 93.760, 94.209, 94.109, 94.079, 94.089]
    x = [i for i in range(1,26)]
    y = [i for i in range(1,26)]
    plt.subplot(1,2,1)
    plt.plot(x, loss_, color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    plt.xticks([0, 5, 10,15, 20, 25])
    plt.title('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(y, acc_train, color='b')
    plt.plot(y, acc_valid, color='r')
    plt.xlabel('epoch')
    plt.ylabel('%') 
    plt.xticks([y for y in range(0,26,5)])
    plt.yticks([y for y in range(20,101,20)])
    plt.title('Accuracy') 
    plt.legend(['Training', 'Testing'],loc='lower right')
    
    plt.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
