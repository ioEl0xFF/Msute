# Mステの入場曲が顔認証を通して流れてくるやつ

## 説明
タイトル嫁

## 参考文献
ほとんどこの方々のコードです。

https://qrunch.net/@Atom/entries/Q9fV8JEIQJVCDVId?ref=qrunch

https://qrunch.net/@Atom/entries/ybUPbsscInQoonto

https://qiita.com/kekeho/items/a0b93695d8a8ac6f1028

## コード一覧

1.face_datamake.py //顔のデータを集め、顔を登録する

2.face_recognition.py //上で登録したデータをもとに顔を識別し、それらを表示する

3.play_sound.py //mp3ファイルを再生する

4.main.py //2〜3のプログラムをループして動かしてるだけ

## 依存関係

opencv-4.4.0

numpy-1.19.2

mutagen-1.45.1

pygame-1.9.6

## 動作環境
```
$cat /etc/os-release
NAME="Arch Linux"
PRETTY_NAME="Arch Linux"
ID=arch
BUILD_ID=rolling
```

## 使い方
顔データの登録
```
$python3 face_datamake.py

 User Id Input <ex:001> >>>  001 //001というIDで登録(000〜005しか対応していない^^)

 Face capture Wait ............

 Face capture End

 Training Start ...

 User 3 trained. Program end
 ```
登録した顔と曲を紐づけ
```
$cp music_file.mp3 ./music_data/1.mp3
$ls ./music_data
1.mp3
```
Mステ
```
$python3 main.py
```
