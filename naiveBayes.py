#coding:UTF-8
import cPickle
import jieba
import ast
import os
from numpy import ndarray


'''
函数功能：将需要分类的数据根据特征集进行向量化
Returns: 向量化的结果
'''
def TextFeatures(data,feature_words):
	data = data.strip()
	data = data.replace(' ','') #去除空格
	data_list = []
	word_cut = jieba.cut(data)
	data_list = list(word_cut)
	def text_features(text, feature_words):		#出现在特征集中，则置1
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	data_feature_list = [text_features(data_list, feature_words)]
	return data_feature_list				#返回结果

def TextClassifing(classifier, data_feature_list):
	result = classifier.predict(data_feature_list)
	result1 = classifier.predict_log_proba(data_feature_list)
	result2 = classifier.predict_proba(data_feature_list)

	return result,result1,result2

with open(os.path.abspath('models/naiveBayes/ad_classifier.pkl'),'rb') as fid:
    classifier = cPickle.load(fid)


#读取feature_words
feature_words_file = open(os.path.abspath('models/naiveBayes/feature_words.txt'),'r')

for line in feature_words_file:
    feature_words = ast.literal_eval(line)




def predict(text):
    data_feature_list = TextFeatures(text,feature_words)
    result,result1,result2 = TextClassifing(classifier,data_feature_list)
    return ndarray.tolist(result)           #<type 'numpy.ndarray'> 返回list
