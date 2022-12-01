from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np

class App:
    def __init__(self):
        root = Tk()
        root.title("Test App")
        root.geometry("1280x720")
        root.maxsize(1600,900)
        """
        lab0 = Label(root, text="Example Interface", background="black", font=("Courier", 35))
        lab0.grid(row=1,column=2)
        lab05 = Text(root, height=2, width=6, bg="gray")
        lab05.grid(row=1,column=1)
        img = cv2.imread('D:/Coding/Retro Trees.jpg', 1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img) 
        lab1 = Label(root, image=imgtk)
        lab1.grid(row=2,column=1, rowspan=3)
        #lab1.pack()
        """
        
        lab0 = Label(root, text="TestText,TexTest", background="black", font= ("Courier", 1))
        #All in one
        lab0.configure(text="Textin Text", background="gray", font= ("Courier", 37))
        lab0.pack()
        lab0["text"] = "We did a lil trickery"
        lab0["bg"] = "red"

        #Auto changes according to the input box
        entr_text = StringVar()
        inp = Entry(root, textvariable=entr_text) 
        print(inp.get())
        inp.pack()
        lab0["textvariable"] = entr_text

        but = Button(root, text="Click Me", command=self.PressBut)
        but.pack()

        frame = Frame(root, width=500, height=300, background="blue")
        frame.bind("<Button-1>", self.LeftClick)
        frame.bind("<Button-2>", self.MiddleClick)
        frame.bind("<Button-3>", self.RightClick)
        frame.pack()
        
        #img = cv2.imread('D:/Coding/a.png', 1)
        img = cv2.imread('D:/Coding/Retro Trees.jpg', 1)

        #b,g,r = cv2.split(img)
        #img = cv2.merge((r,g,b))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im) 
        lab1 = Label(root, image=imgtk)
        lab1.pack()
        
        self.root = root
        self.root.mainloop()

    def PressBut(self):
        print("At your service, User")
        print("Well clicked, son")
    
    def LeftClick(self, event):
        print("Left clicked")

    def RightClick(self, event):
        print("Right clicked")

    def MiddleClick(self, event):
        print("Middle clicked")

app = App()