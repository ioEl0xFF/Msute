from mutagen.mp3 import MP3 as mp3
from inputimeout import inputimeout, TimeoutOccurred
import pygame
import time
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/music_data/'

def play_sound(queue):
  old_filename = ''
  while True:
    id = queue.get()
    if id == 'quit':
      pygame.mixer.music.stop()
      return
    filename = FILE_DIR + id + '.mp3' #再生したいmp3ファイル
    if filename != old_filename:
      #pygame.mixer.music.stop()
      old_filename = filename

      pygame.mixer.init()
      pygame.mixer.music.load(filename) #音源を読み込み
      mp3_length = mp3(filename).info.length #音源の長さ取得
      pygame.mixer.music.play(-1) #再生開始。1の部分を変えるとn回再生(-1で無限ループ)
      #time.sleep(mp3_length + 0.25) #再生開始後、音源の長さだけ待つ(0.25待つのは誤差解消)

