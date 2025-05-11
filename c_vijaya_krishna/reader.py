import os
import numpy as np
from pdf2image import convert_from_path
from typing import List
import pytesseract
import cv2
import  sys
sys.path.append(os.path.join(os.getcwd(), "poppler-23.11.0", "Library", "bin"))
class Output:
    def __init__(self,value:List):
        self.value=value
    def print(self)->None:
        if len(self.value)==1:
            print(self.value[0])
            return
        for i in range(len(self.value)):
            print("\n\nPage No {}\n\n".format(i+1))
            print(self.value[i])
    def get_output(self)->List:
        return self.value
class Reader:
    def __init__(self,POPPLER_PATH=r'poppler-23.11.0\\Library\\bin'):
        assert os.path.exists(POPPLER_PATH)
        self.POPPLER_PATH=POPPLER_PATH
    def read(self,object)->Output:
        if type(object)==str:
            return self.read_from_path(object)
        if type(object)==np.ndarray:
            return self.read_from_tensor(object)
        else:
            raise("Readable Object belongs to pdf images or numpy array")
    def read_from_path(self,path)->Output:
        assert path.split(".")[-1] in ["pdf","jpg","png","jpeg"]
        assert os.path.exists(path)
        output=[]
        if path.endswith(".pdf"):
            images = convert_from_path(path,poppler_path=self.POPPLER_PATH)
            image_arrays = [np.array(img) for img in images]
            for i in image_arrays:
                output.append(pytesseract.image_to_string(i))
        elif path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
            img=cv2.imread(path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            output=[pytesseract.image_to_string(img)]
        return Output(output)
    def read_from_tensor(self,arr)->Output:
        assert len(arr.shape)==3 or len(arr.shape)==2
        output=[pytesseract.image_to_string(arr)]
        return Output(output)

