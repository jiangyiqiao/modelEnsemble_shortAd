#model_Ensemble_Voting_shortAd_Classification
## 微信广告正负样本分类

采用模型融合的方式：投票机制，多数表决原则

## Dependencies
* sk-learn
* tensorflow
* fasttext
* tgrocery
* jieba

## 多模型实现

* [naive Bayes](https://blog.csdn.net/han_xiaoyang/article/details/50629608)
* [word2Vec](https://github.com/jiangyiqiao/word2vec-CNN_shortAd.git)
* [fastText](https://github.com/jiangyiqiao/fastText_shortAd.git)
* [textGrocery](https://github.com/jiangyiqiao/textGrocery_shortAd.git) 
   
正样本：

* xx,xxx,xxxxxxxxx

负样本：

* 微信公众号 AppSo，回复「钱包」看看微信钱包这 6 个秘密使用技巧
 
* 微信号：wszs1981
 
* 长按二维码关注
 
其中，训练集正负样本各1W+，测试集样本各5K+


测试数据集：data/test/ensemble.txt


测试数据为未分词的带标签数据，数据预处理:

    python data/test/pre_progressing.py


测试：
   
    python2 test_models.py

## result

    {'not_ad': 4983, 'ad': 3975}         #预测正确的各个类的数目
    {'not_ad': 5021, 'ad': 4160}         #测试数据集中各个类的数目
    {'not_ad': 5168, 'ad': 4013}         #预测结果中各个类的数目
    not_ad:	 precision:0.964203	 recall:0.992432	 f:0.978114
    ad:	         precision:0.990531	 recall:0.955529	 f:0.972715

