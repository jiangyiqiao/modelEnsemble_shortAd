#encoding:UTF-8
import naiveBayes
import fastTex as fastText
import word2vec as w2v
import textGrocery 

def predict_text(text):
    methods=["naiveBayes","fastText","w2v","textGrocery"]
    results=["0","ad","__label__ad",0]   #0:w2v
    result_ad=0
    result_not_ad=0

    for method in methods:
        if method=="naiveBayes":
            predict=naiveBayes.predict(text)   
            print "naiveBayes predict result:"
            print predict
        elif method=="fastText":
            text = text.decode("utf-8").rstrip()
            predict=fastText.predict(text)
            print "fastTex predict result:"
            print predict
        elif method=="w2v":
            predict=w2v.predict(text)  
            print "word2vec predict result:" 
            print predict

        else:
            predict=textGrocery.predict(text) #type(predict) : <class 'tgrocery.base.GroceryPredictResult'>
            predict=str(predict)
            print "textGrocery predict result:"
            print predict
#判断结果
        if set(predict).issubset(set(results)):
            result_ad+=1
        else:
            result_not_ad+=1
    print("result_ad:",result_ad)
    print("result_not_ad:",result_not_ad)

    result="ad" if result_ad > result_not_ad else "not_ad"

    return result


labels_right = []
labels_predict = []
with open("data/test/ensemble.txt") as fr:
    for line in fr:
        label_right=line.split("\t")[0]
        labels_right.append(label_right)
        text=line.split("\t")[1]
        print text
        label_predict=predict_text(text)
        labels_predict.append(label_predict)
        print ("文本: ")
        print (line)
        print ("真实label: ")
        print (label_right)
        print ("预测label: ")
        print (label_predict)


predict_labels=[]
for predict_label in labels_predict:
    predict_labels.append(predict_label)

A = dict.fromkeys(labels_right,0)   #预测正确的各个类的数目
B = dict.fromkeys(labels_right,0)   #测试数据集中各个类的数目
C = dict.fromkeys(predict_labels,0) #预测结果中各个类的数目
for i in range(0,len(labels_right)):
    B[labels_right[i]] += 1
    C[predict_labels[i]] += 1
    if labels_right[i] == predict_labels[i]:
        A[labels_right[i]] += 1
print (A)
print (B)
print (C)
#计算准确率，召回率，F值
for key in B:
    try:
        r = float(A[key]) / float(B[key])
        p = float(A[key]) / float(C[key])
        f = p * r * 2 / (p + r)
        print ("%s:\t precision:%f\t recall:%f\t f:%f" % (key,p,r,f))
    except:
        print ("error:", key, "right:", A.get(key,0), "real:", B.get(key,0), "predict:",C.get(key,0))



'''
#单个测试
texts=["扫描二维码加关注","很多人喜欢在朋友圈晒车票、护照、飞机票等，但这些票据上的二维码或条形码都含个人姓名、身份证号等信息，借助特殊软件，便能轻易读取"]
methods=["naiveBayes","fastText","w2v","textGrocery"]
results=["0","ad","__label__ad",0]   #0:w2v
for text in texts:
    result_ad=0
    result_not_ad=0

    for method in methods:
        if method=="naiveBayes":
            predict=naiveBayes.predict(text)  
            print type(predict)    
            print "naiveBayes predict result:"
            print predict
            print type(predict)
        elif method=="fastText":
            predict=fastText.predict(text)
            print "fastTex predict result:"
            print predict
            print type(predict)
        elif method=="w2v":
            predict=w2v.predict(text)  
            print "word2vec predict result:" 
            print predict

        else:
            predict=textGrocery.predict(text) #type(predict) : <class 'tgrocery.base.GroceryPredictResult'>
            predict=str(predict)
            print "textGrocery predict result:"
            print predict
            print type(predict)
#判断结果
        if set(predict).issubset(set(results)):
            result_ad+=1
        else:
            result_not_ad+=1
    print("result_ad:",result_ad)
    print("result_not_ad:",result_not_ad)

    result="ad" if result_ad > result_not_ad else "not_ad"

    print(result)
'''


