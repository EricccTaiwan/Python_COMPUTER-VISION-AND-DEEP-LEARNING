import numpy as np
import cv2 
from PyQt5 import QtCore, QtWidgets
import matplotlib.pyplot as plt 
from scipy import signal
from PIL import Image

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1048, 842)
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(60, 50, 951, 741))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(Form)
        self.textEdit_2.setGeometry(QtCore.QRect(100, 80, 191, 631))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(Form)
        self.textEdit_3.setGeometry(QtCore.QRect(320, 80, 191, 631))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(Form)
        self.textEdit_4.setGeometry(QtCore.QRect(540, 80, 191, 631))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(Form)
        self.textEdit_5.setGeometry(QtCore.QRect(760, 80, 191, 631))
        self.textEdit_5.setObjectName("textEdit_5")
        self.one_one = QtWidgets.QPushButton(Form)
        self.one_one.setGeometry(QtCore.QRect(120, 150, 161, 81))
        self.one_one.setObjectName("one_one")
        self.one_one.clicked.connect(one_one_clicked) #1-3

        self.one_two = QtWidgets.QPushButton(Form)
        self.one_two.setGeometry(QtCore.QRect(120, 280, 161, 81))
        self.one_two.setObjectName("one_two")
        self.one_two.clicked.connect(one_two_clicked) #1-2

        self.one_three = QtWidgets.QPushButton(Form)
        self.one_three.setGeometry(QtCore.QRect(120, 410, 161, 81))
        self.one_three.setObjectName("one_three")
        self.one_three.clicked.connect(one_three_clicked) #1-3

        self.one_four = QtWidgets.QPushButton(Form)
        self.one_four.setGeometry(QtCore.QRect(110, 570, 161, 81))
        self.one_four.setObjectName("one_four")
        self.one_four.clicked.connect(one_four_clicked) #1-4

        self.two_one = QtWidgets.QPushButton(Form)
        self.two_one.setGeometry(QtCore.QRect(330, 180, 161, 81))
        self.two_one.setObjectName("two_one")
        self.two_one.clicked.connect(two_one_clicked) #2-1
        
        self.two_two = QtWidgets.QPushButton(Form)
        self.two_two.setGeometry(QtCore.QRect(330, 310, 161, 81))
        self.two_two.setObjectName("two_two")
        self.two_two.clicked.connect(two_two_clicked) #2-2

        self.two_three = QtWidgets.QPushButton(Form)
        self.two_three.setGeometry(QtCore.QRect(330, 440, 161, 81))
        self.two_three.setObjectName("two_three")
        self.two_three.clicked.connect(two_three_clicked) #2-3

        self.three_one = QtWidgets.QPushButton(Form)
        self.three_one.setGeometry(QtCore.QRect(550, 150, 161, 81))
        self.three_one.setObjectName("three_one")
        self.three_one.clicked.connect(three_one_clicked) #3-1
        
        self.three_two = QtWidgets.QPushButton(Form)
        self.three_two.setGeometry(QtCore.QRect(550, 290, 161, 81))
        self.three_two.setObjectName("three_two")
        self.three_two.clicked.connect(three_two_clicked) #3-2

        self.three_three = QtWidgets.QPushButton(Form)
        self.three_three.setGeometry(QtCore.QRect(550, 400, 161, 81))
        self.three_three.setObjectName("three_three")
        self.three_three.clicked.connect(three_three_clicked) #3-3

        self.three_four = QtWidgets.QPushButton(Form)
        self.three_four.setGeometry(QtCore.QRect(550, 530, 161, 81))
        self.three_four.setObjectName("three_four")
        self.three_four.clicked.connect(three_four_clicked) #3-4

        self.four_one = QtWidgets.QPushButton(Form)
        self.four_one.setGeometry(QtCore.QRect(770, 150, 161, 81))
        self.four_one.setObjectName("four_one")
        self.four_one.clicked.connect(four_one_clicked) #4-1

        self.four_two = QtWidgets.QPushButton(Form)
        self.four_two.setGeometry(QtCore.QRect(770, 270, 161, 81))
        self.four_two.setObjectName("four_two")
        self.four_two.clicked.connect(four_two_clicked) #4-2

        self.four_three = QtWidgets.QPushButton(Form)
        self.four_three.setGeometry(QtCore.QRect(780, 400, 161, 81))
        self.four_three.setObjectName("four_three")
        self.four_three.clicked.connect(four_three_clicked) #4-3

        self.four_four = QtWidgets.QPushButton(Form)
        self.four_four.setGeometry(QtCore.QRect(780, 520, 161, 81))
        self.four_four.setObjectName("four_four")
        self.four_four.clicked.connect(four_four_clicked) #4-4

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.textEdit_2.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">1.Image Processing</span></p></body></html>"))
        self.textEdit_3.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">2.Image Smoothing</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p></body></html>"))
        self.textEdit_4.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">3.Edge Detection</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p></body></html>"))
        self.textEdit_5.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">4.Transforms</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p></body></html>"))
        self.one_one.setText(_translate("Form", "1.1Load Image File"))
        self.one_two.setText(_translate("Form", "1.2 Color Separation"))
        self.one_three.setText(_translate("Form", "1.3Color Transformation"))
        self.one_four.setText(_translate("Form", "1.4 Blending"))
        self.two_one.setText(_translate("Form", "2.1 Gaussian blur"))
        self.two_two.setText(_translate("Form", "2.2 Bilaterial filter"))
        self.two_three.setText(_translate("Form", "2.3 Median filter"))
        self.three_one.setText(_translate("Form", "3.1 Gaussian blur"))
        self.three_two.setText(_translate("Form", "3.2 Sobel X"))
        self.three_three.setText(_translate("Form", "3.3 Sobel Y"))
        self.three_four.setText(_translate("Form", "3.4 Magnitude"))
        self.four_one.setText(_translate("Form", "4.1 Resize"))
        self.four_two.setText(_translate("Form", "4.2 Translation"))
        self.four_three.setText(_translate("Form", "4.3 Rotation ,Scaling"))
        self.four_four.setText(_translate("Form", "4.4 Sheering"))

def one_one_clicked():
    path = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg"
    img = cv2.imread(path) 
    cv2.imshow('good',img)
    if __name__ == '__main__':
        size = img.shape
    print('')
    print('Height : ',size[0])
    print('Width : ',size[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def one_two_clicked():
    path = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg"
    img = cv2.imread(path) 
    B,G,R = cv2.split(img);                      
    zeros = np.zeros(img.shape[:2],dtype="uint8")
    cv2.imshow("BLUE",cv2.merge([B,zeros,zeros]))
    cv2.imshow("GREEN",cv2.merge([zeros,G,zeros]))
    cv2.imshow("RED",cv2.merge([zeros,zeros,R]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def one_three_clicked():
    path = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q1_Image\Sun.jpg"
    img = cv2.imread(path) 
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )  
    cv2.imshow('OpenCV function', image)
    B,G,R = cv2.split(img)
    ave = (B//3 + G//3 + R//3)
    cv2.imshow('Averaged weighted',ave)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def one_four_clicked():
    left=0.5
    big_dog = r"C:/Users/Windows/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg"
    small_dog = r"C:/Users/Windows/Desktop/Hw1/Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg"
    def nothing(x):
        pass
    img1 = cv2.imread(big_dog)
    img2 = cv2.imread(small_dog)
    img = np.zeros((500,500,3), np.uint8)
    cv2.namedWindow('Blend')
    cv2.createTrackbar('blending','Blend',0,255,nothing)
    while(1):
        img=cv2.addWeighted(img1,(1.0-left),img2,left,0)
        cv2.imshow('Blend',img)
        left=cv2.getTrackbarPos('blending','Blend')/255
        if(cv2.waitKey(1) == ord('0')):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       
def two_one_clicked():
    hw2=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg"
    image = cv2.imread(hw2)
    image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image1=np.array(image1)
    after = cv2.GaussianBlur(image1, (5, 5), 0)
    plt.figure()
    plt.suptitle('hw2-1')
    plt.subplot(1,2,1)
    plt.imshow(image1)
    plt.subplot(1,2,2)
    plt.title('Gaussian Blur')
    plt.imshow(after)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows

def two_two_clicked():
    hw2=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg"
    image = cv2.imread(hw2)
    image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image1=np.array(image1)
    after = cv2.bilateralFilter(image1, 9,90,90)
    plt.figure()
    plt.suptitle('hw2-1')
    plt.subplot(1,2,1)
    plt.imshow(image1)
    plt.subplot(1,2,2)
    plt.title('Bilateral Filter')
    plt.imshow(after)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows

def two_three_clicked():
    hw2_2=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_pepperSalt.jpg"
    image = cv2.imread(hw2_2)
    image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image3=cv2.medianBlur(image1,3)
    image5=cv2.medianBlur(image1,5)
    plt.figure()
    plt.suptitle('hw2-3')
    plt.subplot(1,3,1)
    plt.imshow(image1)
    plt.subplot(1,3,2)
    plt.title('Median Filter 3x3')
    plt.imshow(image3)
    plt.subplot(1,3,3)
    plt.title('Median Filter 5x5')
    plt.imshow(image5)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows

def three_one_clicked():
    hw3 = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg"
    image_311 = Image.open(hw3).convert("L")
    image_311 = np.asarray(image_311) 
    image_312 = cv2.imread(hw3)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_RGB2BGR)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_BGR2GRAY) 
    x, y = np.mgrid[-1:2, -1:2]
    change = np.exp(-(x**2+y**2))
    change = change/change.sum()
    image_32 = signal.convolve2d(image_312, change, boundary='symm', mode='same')
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image_311, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1,2,2)
    plt.imshow(image_32, cmap=plt.get_cmap('gray'))
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def three_two_clicked():
    hw3 = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg"
    image_312 = cv2.imread(hw3)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_RGB2BGR)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_BGR2GRAY) 
    x, y = np.mgrid[-1:2, -1:2]
    change = np.exp(-(x**2+y**2))
    change = change/change.sum()
    image_312 = signal.convolve2d(image_312, change, boundary='symm', mode='same')
    martix_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    image_32 = signal.convolve2d(image_312,martix_x, boundary='symm', mode='same')
    plt.figure()
    plt.imshow(np.absolute(image_32),cmap='gray', vmin=0, vmax=255)
    plt.show()


def three_three_clicked():
    hw3 = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg"
    image_312 = cv2.imread(hw3)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_RGB2BGR)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_BGR2GRAY) 
    x, y = np.mgrid[-1:2, -1:2]
    change = np.exp(-(x**2+y**2))
    change = change/change.sum()
    image_312 = signal.convolve2d(image_312, change, boundary='symm', mode='same')
    martix_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    image_33 = signal.convolve2d(image_312,martix_y, boundary='symm', mode='same')
    plt.figure()
    plt.imshow(np.absolute(image_33),cmap='gray', vmin=0, vmax=255)
    plt.show()

def three_four_clicked():
    hw3 = r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg"
    image_312 = cv2.imread(hw3)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_RGB2BGR)
    image_312 = cv2.cvtColor(image_312,cv2.COLOR_BGR2GRAY) 
    x, y = np.mgrid[-1:2, -1:2]
    change = np.exp(-(x**2+y**2))
    change = change/change.sum()
    image_312 = signal.convolve2d(image_312, change, boundary='symm', mode='same')
    martix_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    image_32 = signal.convolve2d(image_312,martix_x, boundary='symm', mode='same')
    martix_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    image_33 = signal.convolve2d(image_312,martix_y, boundary='symm', mode='same')
    plt.figure()
    plt.imshow((np.absolute(image_32**2+image_33**2))**0.5,cmap='gray', vmin=0, vmax=255)
    plt.show()

def four_one_clicked():
    hw4=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png"
    img_41 = cv2.imread(hw4)
    img_41=cv2.cvtColor(img_41,cv2.COLOR_RGB2BGR)
    img_41 = cv2.resize(img_41,(256,256))
    plt.imshow(img_41)
    plt.show()

 
def four_two_clicked():
    hw4=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png"
    img_41 = cv2.imread(hw4)
    img_41=cv2.cvtColor(img_41,cv2.COLOR_RGB2BGR)
    img_41 = cv2.resize(img_41,(256,256))
    matrix = np.float32([[1,0,0],[0,1,60]])
    img_42 = cv2.warpAffine(img_41, matrix, (400, 300))
    plt.figure()
    plt.suptitle('After translate')
    plt.imshow(img_42)
    plt.show()

def four_three_clicked():
    hw4=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png"
    img_41 = cv2.imread(hw4)
    img_41=cv2.cvtColor(img_41,cv2.COLOR_RGB2BGR)
    img_41 = cv2.resize(img_41,(256,256))
    matrix = np.float32([[1,0,0],[0,1,60]])
    img_42 = cv2.warpAffine(img_41, matrix, (400, 300))
    matrix = cv2.getRotationMatrix2D((128,188),10,0.5)
    img_43 = cv2.warpAffine(img_42, matrix, (400,300))
    plt.imshow(img_43)
    plt.show()

def four_four_clicked():
    hw4=r"C:\Users\Windows\Desktop\Hw1\Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png"
    img_41 = cv2.imread(hw4)
    img_41=cv2.cvtColor(img_41,cv2.COLOR_RGB2BGR)
    img_41 = cv2.resize(img_41,(256,256))
    matrix = np.float32([[1,0,0],[0,1,60]])
    img_42 = cv2.warpAffine(img_41, matrix, (400, 300))
    matrix = cv2.getRotationMatrix2D((128,188),10,0.5)
    img_43 = cv2.warpAffine(img_42, matrix, (400,300))
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    img_44 = cv2.warpAffine(img_43, M, (400,300))
    plt.imshow(img_44)
    plt.show()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
