# -*- coding: UTF-8 -*-
import cv2
import os
import numpy as np
from PIL import Image

# Current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# 学習画像データ枚数取得変数初期化
sample_cnt = 0

###############################
# VideoCapture用インスタンス生成 #
###############################
cap = cv2.VideoCapture(0)

###############################
# 画像サイズをVGAサイズに変更する  #
###############################
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#######################################################################
# 顔検出を認識する　カスケードファイルは「haarcascade_frontalface_alt2.xml」 #
# カスケードファイルは以下からローカルにダウンロードしておく                    #
# <https://github.com/opencv/opencv/tree/master/data/haarcascades>    #
#######################################################################
face_detector = cv2.CascadeClassifier(CUR_DIR + '/data_xml/haarcascade_frontalface_alt2.xml')

#######################################################
# 学習画像用データから顔認証データymlファイル作成するメソッド  #
# このファイルを顔認証デートのモデルファイルとして使用する     #
#######################################################
def image_learning_make_Labels():

    # リスト保存用変数
    face_list=[]
    ids_list=[]

    # 学習画像データ保存領域パス情報
    path = CUR_DIR + '/image_data'
    # Local Binary Patterns Histogram(LBPH)アルゴリズム　インスタンス
#    recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer=cv2.face_LBPHFaceRecognizer.create()

    # 学習画像ファイルパスを全て取得
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    # 学習画像ファイル分ループ
    for imagePath in imagePaths:

        # グレースケールに変換
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        # UseriDが入っているファイル名からUserID番号として取得
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # 物体認識（顔認識）の実行
        faces = face_detector.detectMultiScale(img_numpy)

        # 認識した顔認識情報を保存
        for (x,y,w,h) in faces:
            face_list.append(img_numpy[y:y+h,x:x+w])
            ids_list.append(id)

    print ("\n Training Start ...")
    ##############################
    # 学習スタート                 #
    ##############################
    recognizer.train(face_list, np.array(ids_list))

    #####################################
    # 学習用した結果を.ymlファイルに保存する  #
    #####################################
    recognizer.save(CUR_DIR + '/trainer/trainer.yml')

    # 学習した顔種類を標準出力
    print("\n User {0} trained. Program end".format(len(np.unique(ids_list))))


#####################################
# 顔認証したい人物の通し番号を入力させる
#####################################
User_id = input('\n User Id Input <ex:001> >>>  ')
print("\n Face capture Wait ............")

####################################
#  学習用画像データ取得と保存
####################################
while(True):

    # カメラで顔データを取得する
    ret, img = cap.read()
    # 画像をグレースケールに変換する
    image_pil = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # NumPyの配列に格納
    gray = np.array(image_pil, 'uint8')
    # Haar-like特徴分類器で顔を検知
    faces = face_detector.detectMultiScale(gray)
    # 学習用画像データを作成
    for (x,y,w,h) in faces:
        # 顔部分を切り取り
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        sample_cnt += 1

        # 画像ファイル名にUSERIDを付与して保存
        cv2.imwrite(CUR_DIR + '/image_data/User.' + str(User_id) + '.' + str(sample_cnt) + '.jpg', image_pil[y:y+h,x:x+w])
        # 画像データを画面表示
        cv2.imshow('image', img)

    # 認証学習画像を10枚
    if sample_cnt >= 10:
         break

print("\n Face capture End ")
########################
# 学習ファイル作成
########################
image_learning_make_Labels()

### カメラ解放 ###
cap.release()
cv2.destroyAllWindows()
