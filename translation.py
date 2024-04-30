import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer
from threading import Thread

class SentenceDisplay(QWidget):
    def __init__(self):
        super().__init__()
        
        self.sentence_label = QLabel()
        self.sentence = ""
        layout = QVBoxLayout()
        layout.addWidget(self.sentence_label)
        self.setLayout(layout)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sentence)
        self.timer.start(1000)  # Update every second
        
        self.setWindowTitle("sentence Display")
        self.setGeometry(100, 100, 400, 400)
        
    def update_sentence(self):
        # Replace this with your actual sentence reading code
        probabilities = np.random.dirichlet(np.ones(26), size=1)
        new_alphabet = translation(probabilities)
        
        self.sentence = self.sentence + str(new_alphabet)
        self.display_sentence()
        
    def display_sentence(self):
        self.sentence = str(self.sentence) + str()
        self.sentence_label.setText(self.sentence)

def translation(prediction):
    '''
    Given the model output, display the predicted alphabet.
    Inputs:
    - prediction: array of dimension 26
    '''
    index = np.argmax(prediction)
    start_char = 'a'
    alphabets = [chr(ord(start_char) + i) for i in range(26)]
    #print(alphabet[index])
    return alphabets[index]

if __name__ == "__main__":
    #t1 = Thread(target=detection)
    #t1.start()
    app = QApplication(sys.argv)
    window = SentenceDisplay()
    window.show()
    sys.exit(app.exec_())
