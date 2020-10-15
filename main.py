import face_recognition
import play_sound
import os
import threading
import queue

# Current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

#  id = face_recognition.face_recognition()
#  play_sound.play_sound(CUR_DIR + '/music_data/' + str(id) + '.mp3')

queue = queue.Queue()
th_fr = threading.Thread(target=face_recognition.face_recognition, args=(queue,))
th_ps = threading.Thread(target=play_sound.play_sound, args=(queue,))

th_fr.start()
th_ps.start()
