import json
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QPushButton,QVBoxLayout,QMessageBox
from PyQt5.QtGui import QPixmap, QKeyEvent
from PyQt5.QtCore import Qt
import glob
import random

class Fenster(QWidget):
    def __init__(self):
        super().__init__()
        self.index=0
        self.imageList=glob.glob("Images/*.jpg")
        self.dict=[]
        self.modeTag=True

        try:
            file=open("Images/tag.json", "r", encoding="utf-8")
            self.dict = json.load(file)
        except:
            pass

        filesDone=[namen["name"] for namen in self.dict]

        imageResList=[x for x in self.imageList if x not in filesDone]

        if(len(imageResList)==0):
            self.modeTag=False
            random.shuffle(self.imageList)

        else:
            self.imageList=imageResList


        layout = QVBoxLayout()
        self.label = QLabel(self)

        self.bntMen = QPushButton("Mann")
        self.btnWoman = QPushButton("Frau")

        layout.addWidget(self.label)
        layout.addWidget(self.bntMen)
        layout.addWidget(self.btnWoman)
        self.setLayout(layout)

        self.bntMen.clicked.connect(lambda: self.click("M"))
        self.btnWoman.clicked.connect(lambda: self.click("W"))

        self.update()

    def update(self):
        self.setWindowTitle("Bildanzeige "+str(len(self.imageList)-self.index))

        pixmap = QPixmap(self.imageList[self.index]) 
        pixmap = pixmap.scaled(
            self.label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.label.update()


    def resizeEvent(self, event):
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_M:
            self.click("M")
        elif event.key() == Qt.Key_W:
            self.click("W")
        elif event.key() == Qt.Key_Escape:
            self.close() 
        else:
            pass

    def click(self,gender):
        if(self.modeTag):
            self.dict.append({"name": self.imageList[self.index],"gender":gender})
        else:
            ele=[d for d in self.dict if d.get("name") == self.imageList[self.index]]
            if (ele[0]["gender"]!=gender):
                msg_box = QMessageBox(text="Zuletzt anders getaggt "+self.imageList[self.index])
                msg_box.exec_()
                return

            
        
        self.index+=1
        if(self.index==len(self.imageList)): self.close()
        self.update()


    def closeEvent(self, event):
        self.close()

    def close(self):
        with open("Images/tag.json", "w", encoding="utf-8") as datei:
            json.dump(self.dict, datei, indent=4, ensure_ascii=False)
        super().close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenster = Fenster()
    fenster.show()
    fenster.setGeometry(400,400,800,1500)
    sys.exit(app.exec_())