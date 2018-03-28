# coding: utf-8

from tgrocery import Grocery

grocery = Grocery('models/textGrocery')
#加载模型
grocery.load()

def predict(text):
    text = text.decode("utf-8").rstrip()
    result=grocery.predict(text)
    return result

