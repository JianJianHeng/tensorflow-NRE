#复现2016年ACL的NRE相关论文
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import math
import sys
import time
import init
import CNN_ATT_test as test

binFilePath='data/vec.bin'
vecFilePath='data/RE/word2vec1_01.txt'
REFilePath='data/RE/relation2id.txt'
trainFilePath='data/RE/train.txt'
testFilePath='data/RE/test.txt'
modelDirectoryPath='model/'
modelSavePath=modelDirectoryPath+'my_model.h5'
logFile='log/log.txt'
trainLogFile='log/train_log_7_19.txt'

# #embedding
# emb_word_dim=50
# dimensionWPE=5    #position embedding大小
# #CNN encoder
# dimensionC=230    #卷积核
# window=3          #滑动窗口大小
#
# #train
# dropoutRate=0.5
# batchSize=50
# learningRate=0.0005
# NA_relation_proportion=0.93   #训练批次中NA关系的比例
# nPoch=5
# #test
# predictThreshold=0.4    #当预测的概率大于该阈值时判断存在关系
#
# TEST_NA_relation_proportion=0.98


# class dataLoader():
#     def __init__(self):
#         self.limit=30      #word与head和tail的最大距离限制
#         #表示与E1实体和E2实体的最大距离和最小距离
#         self.PositionMinE1=0
#         self.PositionMaxE1=0
#         self.PositionTotalE1=0
#         self.PositionMinE2=0
#         self.PositionMaxE2=0
#         self.PositionTotalE2=0
#
#         #读取word2vec信息
#         with open(binFilePath,'rb') as f:
#             context=f.readline().decode('utf-8').split(' ')
#         self.wordTotal=int(context[0])
#         self.wordTotal+=1
#         self.dimension=int(context[1])
#         print("wordTotal:\t",self.wordTotal,"\tdimension:\t",self.dimension)
#
#         self.wordVec=np.zeros((self.wordTotal,self.dimension),dtype=np.float32)   #word2Vec矩阵
#         self.wordList={}                                                        #word字典
#         self.wordMapping={}
#         with open(vecFilePath,'r',encoding='utf-8') as f:
#             wvec=f.readline().split(' ')
#             while wvec[0]:
#                 for i in range(self.dimension):
#                     self.wordVec[len(self.wordList)][i]=float(wvec[i+1])
#                 self.wordMapping[wvec[0]]=len(self.wordList)
#                 self.wordList[len(self.wordList)]=wvec[0]
#                 wvec=f.readline().split(' ')
#
#         #读取关系数据
#         self.relationMapping = {}
#         self.nam={}
#         with open(REFilePath,'r') as f:
#             relation=f.readline().split(' ')
#             while relation[0]:
#                 self.relationMapping[relation[0]]=int(relation[1])
#                 self.nam[int(relation[1])] = relation[0]
#                 relation=f.readline().split(' ')
#         # for rel,id in self.relationMapping.items():
#         #     print(rel)
#         self.relationTotal=len(self.nam)
#
#         #读取训练数据
#         self.bagsTrain={}     #多示例训练数据集
#         self.bagsTrain_NA={}  #NA关系训练数据集
#         self.bagsTrain_notNA={}  #非NA关系训练数据集
#         self.headList={}      #e1列表
#         self.tailList={}      #e2列表
#         self.relationList={}  #关系列表(标签)
#         self.trainLength={}   #记录训练的句子长度
#
#         # 计算每个word与head和tail的距离
#         self.trainList = {}
#         self.trainPositionE1 = {}
#         self.trainPositionE2 = {}
#         count=0               #记录有多少个训练数据
#         with open(trainFilePath,'r') as f:
#             trainData=f.readline().split('\t')
#             while trainData[0]:
#                 #读取实体数据
#                 head_s=trainData[2]
#                 if head_s in self.wordMapping.keys():
#                     head = self.wordMapping[head_s]
#                 else:  # 若不存在词典中，标记为UNK
#                     head = self.wordMapping['UNK']
#                 tail_s=trainData[3]
#                 if tail_s in self.wordMapping.keys():
#                     tail = self.wordMapping[tail_s]
#                 else:
#                     tail = self.wordMapping['UNK']
#                 rel=trainData[4]
#                 sen=trainData[5].split(' ')
#                 #构造训练数据
#                 if head_s+'\t'+tail_s+'\t'+rel in self.bagsTrain.keys():
#                     self.bagsTrain[head_s+'\t'+tail_s+'\t'+rel].append(count)
#                 else:
#                     self.bagsTrain[head_s+'\t'+tail_s+'\t'+rel]=[]
#                     self.bagsTrain[head_s + '\t' + tail_s + '\t' + rel].append(count)
#
#
#                 #读取句子
#                 leng=0         #句子长度
#                 lefnum=0       #head的位置
#                 rignum=0       #tail的位置
#                 tmpp={}        #将word转换成id后连接成序列
#                 for word in sen:
#                     if word is '###END###':
#                         break
#                     if word in self.wordMapping.keys():
#                         gg=self.wordMapping[word]
#                     else:
#                         gg=self.wordMapping['UNK']
#                     if gg==head:
#                         lefnum=leng
#                     if gg==tail:
#                         rignum=leng
#                     tmpp[leng]=gg
#                     leng += 1
#
#                 #将句子中的信息加入训练数据中
#                 try:
#                     self.relationList[count]=self.relationMapping[rel]
#                 except:
#                     #将文本未收集到的关系加入relationMapping中
#                     self.relationMapping[rel]=self.relationTotal
#                     self.nam[self.relationTotal] = rel
#                     self.relationTotal += 1
#                     self.relationList[count] = self.relationMapping[rel]
#
#                 self.headList[count]=head
#                 self.tailList[count]=tail
#                 self.trainLength[count]=leng
#
#                 #构造NA和notNA的数据集
#                 if self.relationList[count]==0:
#                     if head_s + '\t' + tail_s + '\t' + rel in self.bagsTrain_NA.keys():
#                         self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel].append(count)
#                     else:
#                         self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel] = []
#                         self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel].append(count)
#                 else:
#                     if head_s + '\t' + tail_s + '\t' + rel in self.bagsTrain_notNA.keys():
#                         self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel].append(count)
#                     else:
#                         self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel] = []
#                         self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel].append(count)
#
#                 con = []      #con是训练数据
#                 conl = []   #conl是与head的距离
#                 conr = []   #conr是与tail的距离
#                 for i in range(leng):
#                     con.append(tmpp[i])
#                     conl.append(lefnum - i)
#                     conr.append(rignum - i)
#                     if conl[i] >= self.limit:
#                         conl[i] = self.limit
#                     if (conr[i] >= self.limit):
#                         conr[i] = self.limit
#                     if (conl[i] <= -self.limit):
#                         conl[i] = -self.limit
#                     if (conr[i] <= -self.limit):
#                         conr[i] = -self.limit
#                     if (conl[i] > self.PositionMaxE1):
#                         self.PositionMaxE1 = conl[i]
#                     if (conr[i] > self.PositionMaxE2):
#                         self.PositionMaxE2 = conr[i]
#                     if (conl[i] < self.PositionMinE1):
#                         self.PositionMinE1 = conl[i]
#                     if (conr[i] < self.PositionMinE2):
#                         self.PositionMinE2 = conr[i]
#
#                 self.trainList[count]=con
#                 self.trainPositionE1[count]=conl
#                 self.trainPositionE2[count]=conr
#
#                 count+=1
#                 trainData = f.readline().split('\t')
#
#         # with open(logFile,'w+') as f:
#
#             # print("----------------bagsTrain-----------------",file=f)
#             # for text, id in self.bagsTrain.items():
#             #     print(text, id,file=f)
#
#             # print("----------------headList-----------------",file=f)
#             # for text, id in self.headList.items():
#             #     print(text, id,file=f)
#
#             # print("----------------tailList-----------------",file=f)
#             # for text, id in self.tailList.items():
#             #     print(text, id,file=f)
#
#             # print("----------------trainLength-----------------", file=f)
#             # for text, id in self.trainLength.items():
#             #     print(text, id, file=f)
#
#             # print("----------------trainList-----------------",file=f)
#             # for text, id in self.trainList.items():
#             #     print(text, id,file=f)
#             #
#             # print("----------------trainPositionE1-----------------", file=f)
#             # for text, id in self.trainPositionE1.items():
#             #     print(text, id, file=f)
#             #
#             # print("----------------trainPositionE2-----------------", file=f)
#             # for text, id in self.trainPositionE2.items():
#             #     print(text, id, file=f)
#
#         # 读取测试数据
#         self.bagsTest = {}  # 测试数据集
#         #self.bagsTest_notNA={}  #非NA的数据集
#         self.tot=0   #非NA关系的bags数，用于召回率计算使用
#         self.bagsTest_Label = {}  # 最终所有bags正确的列表
#
#         self.testHeadList = {}  # e1列表
#         self.testTailList = {}  # e2列表
#         self.testRelationList = {}  # 关系列表(标签)
#         self.testTrainLength = {}  # 记录测试的句子长度
#
#         # 计算每个word与head和tail的距离
#         self.testTrainList = {}
#         self.testPositionE1 = {}
#         self.testPositionE2 = {}
#         count = 0  # 记录有多少个测试数据
#         with open(testFilePath, 'r') as f:
#             trainData = f.readline().split('\t')
#             while trainData[0]:
#                 # 读取实体数据
#                 head_s = trainData[2]
#                 if head_s in self.wordMapping.keys():
#                     head = self.wordMapping[head_s]
#                 else:  # 若不存在词典中，标记为UNK
#                     head = self.wordMapping['UNK']
#                 tail_s = trainData[3]
#                 if tail_s in self.wordMapping.keys():
#                     tail = self.wordMapping[tail_s]
#                 else:
#                     tail = self.wordMapping['UNK']
#                 rel = trainData[4]
#                 sen = trainData[5].split(' ')
#                 #用于预测的测试包数据中字典键不包含关系rel
#                 if head_s+'\t'+tail_s in self.bagsTest.keys():
#                     self.bagsTest[head_s+'\t'+tail_s].append(count)
#                 else:
#                     self.bagsTest[head_s+'\t'+tail_s]=[]
#                     self.bagsTest[head_s + '\t' + tail_s].append(count)
#
#
#
#                 # 读取句子
#                 leng = 0  # 句子长度
#                 lefnum = 0  # head的位置
#                 rignum = 0  # tail的位置
#                 tmpp = {}  # 将word转换成id后连接成序列
#                 for word in sen:
#                     if word is '###END###':
#                         break
#                     if word in self.wordMapping.keys():
#                         gg = self.wordMapping[word]
#                     else:
#                         gg = self.wordMapping['UNK']
#                     if gg == head:
#                         lefnum = leng
#                     if gg == tail:
#                         rignum = leng
#                     tmpp[leng] = gg
#                     leng += 1
#
#                 # 将句子中的信息加入测试数据中
#                 self.testRelationList[count] = self.relationMapping[rel]
#                 try:
#                     self.testRelationList[count] = self.relationMapping[rel]
#                 except:
#                     # 将文本未收集到的关系加入relationMapping中
#                     self.relationMapping[rel] = self.relationTotal
#                     self.nam[self.relationTotal] = rel
#                     self.relationTotal += 1
#                     self.testRelationList[count] = self.relationMapping[rel]
#                 #构建最终的label数据
#                 if head_s+'\t'+tail_s+'\t'+rel in self.bagsTest_Label.keys():
#                     self.bagsTest_Label[head_s+'\t'+tail_s+'\t'+rel].append(count)
#                 else:
#                     self.bagsTest_Label[head_s+'\t'+tail_s+'\t'+rel]=[]
#                     self.bagsTest_Label[head_s + '\t' + tail_s+'\t'+rel].append(count)
#                     # 统计正样本
#                     if self.relationMapping[rel] != 0:
#                         self.tot += 1
#
#                 self.testHeadList[count] = head
#                 self.testTailList[count] = tail
#                 self.testTrainLength[count] = leng
#
#                 con = []  # con是测试数据
#                 conl = []  # conl是与head的距离
#                 conr = []  # conr是与tail的距离
#                 for i in range(leng):
#                     con.append(tmpp[i])
#                     conl.append(lefnum - i)
#                     conr.append(rignum - i)
#                     if conl[i] >= self.limit:
#                         conl[i] = self.limit
#                     if (conr[i] >= self.limit):
#                         conr[i] = self.limit
#                     if (conl[i] <= -self.limit):
#                         conl[i] = -self.limit
#                     if (conr[i] <= -self.limit):
#                         conr[i] = -self.limit
#                     if (conl[i] > self.PositionMaxE1):
#                         self.PositionMaxE1 = conl[i]
#                     if (conr[i] > self.PositionMaxE2):
#                         self.PositionMaxE2 = conr[i]
#                     if (conl[i] < self.PositionMinE1):
#                         self.PositionMinE1 = conl[i]
#                     if (conr[i] < self.PositionMinE2):
#                         self.PositionMinE2 = conr[i]
#
#                 self.testTrainList[count] = con
#                 self.testPositionE1[count] = conl
#                 self.testPositionE2[count] = conr
#
#                 count += 1
#                 trainData = f.readline().split('\t')
#
#         # with open(logFile,'w+') as f:
#         #
#         #     print("----------------bagsTest-----------------",file=f)
#         #     for text, id in self.bagsTest.items():
#         #         print(text, id,file=f)
#         #
#         #     print("----------------testHeadList-----------------",file=f)
#         #     for text, id in self.testHeadList.items():
#         #         print(text, id,file=f)
#         #
#         #     print("----------------testTailList-----------------",file=f)
#         #     for text, id in self.testTailList.items():
#         #         print(text, id,file=f)
#         #
#         #     print("----------------testTrainLength-----------------", file=f)
#         #     for text, id in self.testTrainLength.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------testTrainList-----------------",file=f)
#         #     for text, id in self.testTrainList.items():
#         #         print(text, id,file=f)
#         #
#         #     print("----------------testPositionE1-----------------", file=f)
#         #     for text, id in self.testPositionE1.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------testPositionE2-----------------", file=f)
#         #     for text, id in self.testPositionE2.items():
#         #         print(text, id, file=f)
#
#         self.bagsTest_notNA = {}  # 测试包中存在非NA关系的bags
#         self.bagsTest_NA = {}  # 测试包中关系全为NA的bags
#         for bagsName,testList in self.bagsTest.items():
#             #若包中存在不为NA关系的元素
#             list=[]
#             for index in testList:
#                 list.append(self.testRelationList[index])
#             # print('testList:',testList)
#             # print('list:',list)
#             if np.all(np.array(list)==0):
#                 self.bagsTest_NA[bagsName]=testList
#             else:
#                 self.bagsTest_notNA[bagsName]=testList
#
#         #处理训练和测试的PositionE,将距离全部变成正数
#         for i in range(len(self.trainPositionE1)):
#             leng=self.trainLength[i]
#             for j in range(leng):
#                 self.trainPositionE1[i][j]=self.trainPositionE1[i][j]-self.PositionMinE1
#             for j in range(leng):
#                 self.trainPositionE2[i][j]=self.trainPositionE2[i][j]-self.PositionMinE2
#
#         for i in range(len(self.testPositionE1)):
#             leng=self.testTrainLength[i]
#             for j in range(leng):
#                 self.testPositionE1[i][j]=self.testPositionE1[i][j]-self.PositionMinE1
#             for j in range(leng):
#                 self.testPositionE2[i][j]=self.testPositionE2[i][j]-self.PositionMinE2
#
#         # with open(logFile,'w+') as f:
#         #     print("----------------bagsTrain-----------------", file=f)
#         #     for text, id in self.bagsTrain.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------bagsTest-----------------", file=f)
#         #     for text, id in self.bagsTest.items():
#         #         print(text, id, file=f)
#
#         #     print("----------------trainPositionE1-----------------", file=f)
#         #     for text, id in self.trainPositionE1.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------trainPositionE2-----------------", file=f)
#         #     for text, id in self.trainPositionE2.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------testPositionE1-----------------", file=f)
#         #     for text, id in self.testPositionE1.items():
#         #         print(text, id, file=f)
#         #
#         #     print("----------------testPositionE2-----------------", file=f)
#         #     for text, id in self.testPositionE2.items():
#         #         print(text, id, file=f)
#
#         print("trainList:",len(self.trainList))
#         print("bagsTrain:", len(self.bagsTrain))
#         print('bagsTrain_NA:',len(self.bagsTrain_NA))
#         print('bagsTrain_notNA:',len(self.bagsTrain_notNA))
#
#         print("testTrainList:", len(self.testTrainList))
#         print('bagsTest:', len(self.bagsTest))
#         print('bagsTest_Label:', len(self.bagsTest_Label))
#         print('tot:', self.tot)
#         print('bagsTest_NA:',len(self.bagsTest_NA))
#         print('bagsTest_notNA:', len(self.bagsTest_notNA))
#
#         print("relationTotal:\t", self.relationTotal)
#         print(self.PositionMinE1,' ',self.PositionMaxE1,' ',self.PositionMinE2,' ',self.PositionMaxE2)
#         self.PositionTotalE1=self.PositionMaxE1-self.PositionMinE1+1
#         self.PositionTotalE2 = self.PositionMaxE2 - self.PositionMinE2 + 1
#
#         print("Init end...")
#
#     def get_train_batch(self,n):
#         bagsBatch=[]
#         relationBatch=[]
#         #数据集中NA关系和not_NA关系的个数
#         NA_num=int(n*NA_relation_proportion)
#         not_NA_num=n-NA_num
#         #添加NA_num个NA关系和not_NA_num个非NA关系的数据
#         for i in range(NA_num):
#             bx=random.choice(list(self.bagsTrain_NA))
#             bagsBatch.append(bx)
#             relationBatch.append(0)
#         for i in range(not_NA_num):
#             bx = random.choice(list(self.bagsTrain_notNA))
#             by = self.relationList[self.bagsTrain[bx][0]]
#             bagsBatch.append(bx)
#             relationBatch.append(by)
#
#         return bagsBatch,relationBatch
#
#     #批量获取测试数据
#     def get_test_batch(self, n):
#         testBags=[]
#         testLabels=[]
#         # 测试集中NA关系和not_NA关系的个数
#         NA_num = int(n * TEST_NA_relation_proportion)
#         not_NA_num = n - NA_num
#         testBags_NA=random.sample(list(self.bagsTest_NA),NA_num)
#         # print('testBags1:',testBags)
#         # print('testBags1:', type(testBags))
#         for bx in testBags_NA:
#             testBags.append(bx)
#             by=set()
#             for index in self.bagsTest[bx]:
#                 by.add(self.testRelationList[index])
#             testLabels.append(by)
#         testBags_notNA = random.sample(list(self.bagsTest_notNA), not_NA_num)
#         for bx in testBags_notNA:
#             testBags.append(bx)
#             by=set()
#             for index in self.bagsTest[bx]:
#                 by.add(self.testRelationList[index])
#             testLabels.append(by)
#
#         test_data=zip(testBags,testLabels)
#         test_data = list(test_data)
#         random.shuffle(test_data)
#         # print('test_data:',test_data)
#         test_bags = []
#         test_labels = []
#         test_bags[:], test_labels[:] = zip(*test_data)
#         return test_bags,test_labels
#
#     #将所有测试数据打包返回
#     def get_test_all_batch(self):
#         testBags = list(self.bagsTest.keys())
#         testLabels = []
#         #print('testBags:',len(testBags))
#         #print('testBags:', type(testBags))
#         for bx in testBags:
#             by=set()
#             for index in self.bagsTest[bx]:
#                 by.add(self.testRelationList[index])
#             testLabels.append(by)
#
#         test_data = zip(testBags, testLabels)
#         test_data = list(test_data)
#         random.shuffle(test_data)
#         test_bags = []
#         test_labels = []
#         test_bags[:], test_labels[:] = zip(*test_data)
#         return test_bags, test_labels

# def att_W_init(shape, dtype=np.float32):
#     return np.identity(shape[0])
#
# def enc_V_init(shape,dtype=np.float32):
#     return data_loader.wordVec



class CNN_Selective_Attention(keras.Model):
    def __init__(self,wordTotal,PositionTotalE1,PositionTotalE2,dimension,dimensionWPE,dimensionC,window,relationTotal,
                 data_loader,dropoutRate,learning_rate,batchSize,predictThreshold=0.75):
        super().__init__()
        self.dimension=dimension
        self.dimensionC=dimensionC
        self.dimensionWPE=dimensionWPE
        self.window=window
        self.data_loader=data_loader
        self.learning_rate=learning_rate
        self.predictThreshold=predictThreshold

        self.con=math.sqrt(6.0/(dimensionC+relationTotal))
        self.con1=math.sqrt(6.0/((dimensionWPE+dimension)*window))

        #tip列表(此处初始化无作用)
        self.tip=np.zeros((1,self.dimensionC),dtype=np.int32)

        #手动设置输入
        #self._set_inputs(tf.TensorSpec([None,self.dimension],tf.float32,name='inputs'))
        #embedding
        self.enc_v_embedding=keras.layers.Embedding(
            input_dim=wordTotal,output_dim=dimension,embeddings_initializer=init.enc_V_init
        )  #输入[seq_len] 输出[seq_len,dimension]
        self.enc_p1_embedding=keras.layers.Embedding(
            input_dim=PositionTotalE1,output_dim=dimensionWPE,embeddings_initializer=tf.initializers.RandomNormal(-self.con1,self.con1)
        )  #输入[seq_len] 输出[seq_len,dimensionWPE]
        self.enc_p2_embedding = keras.layers.Embedding(
            input_dim=PositionTotalE2, output_dim=dimensionWPE,
            embeddings_initializer=tf.initializers.RandomNormal(-self.con1, self.con1)
        )  # 输入[seq_len] 输出[seq_len,dimensionWPE]

        #encoder层
        self.conv2d_v=keras.layers.Conv2D(
            filters=dimensionC,kernel_size=(window,dimension),padding='valid',
        ) #input:[seq_len,dimension] output:[seq_len-windows+1,dimensionC]
        self.conv2d_p1 = keras.layers.Conv2D(
            filters=dimensionC, kernel_size=(window, dimensionWPE), padding='valid',
        )#input:[seq_len,1] output:[seq_len-windows+1,dimensionWPE]
        self.conv2d_p2 = keras.layers.Conv2D(
            filters=dimensionC, kernel_size=(window, dimensionWPE), padding='valid',
        )#input:[seq_len,1] output:[seq_len-windows+1,dimensionWPE]

        #selective_attention
        #attention的对角矩阵
        self.att_W=[keras.layers.Dense(units=dimensionC,use_bias=False,kernel_initializer=init.att_W_init) for n in range(self.data_loader.relationTotal)]

        #query vector
        self.matrixRelation=keras.layers.Dense(units=self.data_loader.relationTotal,kernel_initializer='RandomNormal')

        #dropout
        self.dropout=keras.layers.Dropout(dropoutRate,input_shape=(dimensionC,))

        #optimizer
        self.opt=keras.optimizers.SGD(learning_rate=self.learning_rate)

        #表示是否是第一次训练
        self.flag=True

    #设置是否是第一次训练
    def set_flag(self,flag):
        self.flag=flag

    # def set_enc_v_embedding(self):
    #     self.enc_v_embedding.set_weights([enc_V_init(None)])

    def encode(self,sen,p1,p2,index,training=True):
        embedded_v=self.enc_v_embedding(sen)   #[seq_len,dimension]
        embedded_p1=self.enc_p1_embedding(p1)  #[seq_len,dimensionWPE]
        embedded_p2=self.enc_p2_embedding(p2)  #[seq_len,dimensionWPE]

        #扩维
        embedded_v=tf.expand_dims(embedded_v,axis=0)
        embedded_v = tf.expand_dims(embedded_v, axis=3)  #[1,seq_len,dimension,1]
        embedded_p1=tf.expand_dims(embedded_p1,axis=0)
        embedded_p1 = tf.expand_dims(embedded_p1, axis=3)  #[1,seq_len,dimensionWPE,1]
        embedded_p2=tf.expand_dims(embedded_p2,axis=0)
        embedded_p2 = tf.expand_dims(embedded_p2, axis=3)  #[1,seq_len,dimensionWPE,1]

        c_v=self.conv2d_v(embedded_v)
        c_p1=self.conv2d_p1(embedded_p1)
        c_p2=self.conv2d_p2(embedded_p2)

        c=c_v+c_p1+c_p2               #[1,seq_len-window+1,1,230]
        c=tf.squeeze(c,axis=[0,2])    #[seq_len-window+1,230]
        if training:
            # 记录最大池化的位置
            self.tip[index]=tf.argmax(c,axis=0)  #[1,230]
        # print('tip:', self.tip.shape)
        # print('tip:',self.tip)
        #最大池化
        c=tf.reduce_max(c,axis=0)     #[1,230]
        #激活
        c=tf.tanh(c)                  #[1,230]

        return c

    def selective_attention(self,cnn_output,rel,training=True):
        #若是第一次训练，生成所有的att_W矩阵
        if self.flag:
            self.flag=False
            for i in range(self.data_loader.relationTotal):
                self.att_W[i](cnn_output)

        #attention的权重
        weight=self.att_W[rel](cnn_output)       #[sen_n,dimensionC]
        weight=self.matrixRelation(weight)  #[sen_n,relationTotal]

        weight=tf.exp(weight[:,rel])
        weight_sum=tf.reduce_sum(weight)
        weight=tf.reshape(weight/weight_sum,(weight.shape[0],1))

        att_output = weight * cnn_output  # [sen_n,230]
        att_output = tf.reduce_sum(att_output, axis=0)  # [1,230]
        att_output = self.dropout(att_output, training=training)  # [1,230]
        att_output = tf.reshape(att_output, (1, self.dimensionC))  # [1,230]

        output = self.matrixRelation(att_output)  # [1,58]
        output = tf.nn.softmax(output)  # [1,58]

        return output,att_output,weight


    def call(self,inputs):
        #yTemp=np.zeros((len(inputs),self.data_loader.relationTotal),dtype=np.float32)
        #yTemp=tf.constant(tf.zeros([len(inputs),self.data_loader.relationTotal]))
        #print('yTemp:',yTemp.shape)
        #count=0
        bagsName=inputs
        bagsSize=len(self.data_loader.bagsTrain[bagsName])
        #rList=[]
        rList=np.zeros((bagsSize,self.dimensionC),dtype=np.float32)
        rel = self.data_loader.relationList[self.data_loader.bagsTrain[bagsName][0]]
        self.tip=np.zeros((bagsSize,self.dimensionC),dtype=np.int32)      #初始化tip
        for i in range(bagsSize):
            index=self.data_loader.bagsTrain[bagsName][i]
            rel1=self.data_loader.relationList[index]
            if rel1!=rel:
                print('数据处理错误...')
                sys.exit(1)
            rList[i,:]=self.encode(np.array(self.data_loader.trainList[index]),np.array(self.data_loader.trainPositionE1[index]),np.array(self.data_loader.trainPositionE2[index]),i)

        #cnn_output=np.array(rList)                         #[sen_n,230]
        cnn_output=rList
        output,att_output,weight=self.selective_attention(cnn_output,rel)    #[sen_n,1]
        # att_output=weight*cnn_output                       #[sen_n,230]
        # att_output=tf.reduce_sum(att_output,axis=0)        #[1,230]
        # att_output=self.dropout(att_output,training=True)  #[1,230]
        # att_output=tf.reshape(att_output,(1,dimensionC))   #[1,230]
        #
        # output=self.matrixRelation(att_output)             #[1,58]
        # output=tf.nn.softmax(output)                       #[1,58]
        #print('output type:',type(output))
        #output=tf.reshape(output,(output.shape[1],))
        #print('output:',output.shape)

        #yTemp[count,:]=output                               #[1,58]
        #print('yTemp type:',type(yTemp))
        #count+=1

        #print(yTemp)
        #y_pred=np.array(yTemp)
        return output,att_output,cnn_output,weight

    #预测函数
    def inference(self,bagsName):
        bagsSize = len(self.data_loader.bagsTest[bagsName])
        rList = np.zeros((bagsSize, self.dimensionC), dtype=np.float32)
        for i in range(bagsSize):
            index=self.data_loader.bagsTest[bagsName][i]
            rList[i,:]=self.encode(np.array(self.data_loader.testTrainList[index]),np.array(self.data_loader.testPositionE1[index]),np.array(self.data_loader.testPositionE2[index]),i,training=False)
        cnn_output = rList

        #测试时经过所有的relation对角矩阵
        relation_num=self.data_loader.relationTotal
        all_output=np.zeros((relation_num,relation_num))     #[58,58]

        for i in range(relation_num):
            output,_,_=self.selective_attention(cnn_output,i,training=False)
            all_output[i]=output[0]

        result=tf.reduce_max(all_output,axis=0).numpy()  #[58,]

        predict=result>self.predictThreshold
        predict=np.array(np.where(predict))

        #若预测值全都小于阈值
        if predict.size==0:
            #方法一：输出最大概率的值
            #predict=np.argmax(result)
            #predict=predict.reshape((1,1))
            #方法二：不确定的全部输出为NA关系
            predict=np.zeros((1,1),dtype=np.float32)

        return predict.reshape((-1,))

    #将梯度传到encoder中
    def train_gradient(self,index,r,tip,grad):
        sentence=np.array(self.data_loader.trainList[index])      #[seq_len,]
        trainPositionE1=np.array(self.data_loader.trainPositionE1[index])   #[seq_len.]
        trainPositionE2 =np.array(self.data_loader.trainPositionE2[index])  #[seq_len]
        #length=self.data_loader.trainLength[index]
        #e1=self.data_loader.headList[index]
        #e2=self.data_loader.tailList[index]
        #r1=self.data_loader.relationList[index]
        #alpha=self.learning_rate

        #获得权重
        matrixW1,matrixB1=self.conv2d_v.get_weights()         #[3,50,1,230]  [230,]
        matrixW1Dao=matrixW1.copy()
        matrixW1PositionE1,matrixW1PositionB1=self.conv2d_p1.get_weights()    #[3,5,1,230]
        matrixW1PositionE1Dao=matrixW1PositionE1.copy()
        matrixW1PositionE2,matrixW1PositionB2=self.conv2d_p2.get_weights()    #[3,5,1,230]
        matrixW1PositionE2Dao=matrixW1PositionE2.copy()
        wordVec=self.enc_v_embedding.get_weights()[0]         #[114043,50]
        wordVecDao=wordVec.copy()
        positionVecE1 = self.enc_p1_embedding.get_weights()[0] #[61,5]
        positionVecE1Dao=positionVecE1.copy()
        positionVecE2=self.enc_p2_embedding.get_weights()[0]   #[61,5]
        positionVecE2Dao=positionVecE2.copy()

        # print('r:',r.shape)  [230]
        g1=grad*(1-tf.square(r))  #[230,]
        matrixB1 -= g1
        g1=tf.reshape(g1,(g1.shape[0],1))  #[230,1]
        #print('g1:',g1.shape)
        # print('sentence:',sentence)
        # print('tip:',tip)

        # numpy方法
        for j in range(self.window):
            #wordVec的下标
            position_v = sentence[tip+j]     #[230,]
            position_e1=trainPositionE1[tip+j]
            position_e2=trainPositionE2[tip+j]

            #求梯度
            #g_mW1=(g1*wordVecDao[position_v]).numpy()                   #[230,50]
            g_mW1=tf.expand_dims((g1*wordVecDao[position_v]).numpy().T,axis=1)         #[50,1,230]
            #g_mE1=(g1*positionVecE1Dao[position_e1]).numpy()
            g_mE1=tf.expand_dims((g1*positionVecE1Dao[position_e1]).numpy().T,axis=1)         #[5,1,230]
            #g_mE2 = (g1 * positionVecE2Dao[position_e2]).numpy()
            g_mE2 = tf.expand_dims((g1 *positionVecE2Dao[position_e2]).numpy().T, axis=1)  # [5,1,230]
            matrixW1[j]-=g_mW1
            matrixW1PositionE1[j]-=g_mE1
            matrixW1PositionE2[j]-=g_mE2

            #g_mWV=tf.squeeze(matrixW1Dao[j],axis=1)        #[50,230]
            g_mWV=tf.squeeze(matrixW1Dao[j],axis=1).numpy().T                          #[230,50]
            g_mWV=(g1*g_mWV).numpy()                 #[230,50]
            #g_pV1=tf.squeeze(matrixW1PositionE1Dao[j],axis=1)
            g_pV1 = tf.squeeze(matrixW1PositionE1Dao[j],axis=1).numpy().T       # [230,5]
            g_pV1 = (g1 * g_pV1).numpy()  # [230,5]
            #g_pV2 = tf.squeeze(matrixW1PositionE2Dao[j], axis=1)
            g_pV2 = tf.squeeze(matrixW1PositionE2Dao[j], axis=1).numpy().T  # [230,5]
            g_pV2 = (g1 * g_pV2).numpy()  # [230,5]
            wordVec[position_v]-=g_mWV
            positionVecE1[position_e1]-=g_pV1
            positionVecE2[position_e2]-=g_pV2

            # print('wordVec:',wordVec[position_v[0]])
            # print('wordVecDao:',wordVecDao[position_v[0]])
            # print('matrixW1:',matrixW1[j,0,0,0])
            # print('matrixW1Dao:', matrixW1Dao[j, 0, 0, 0])
            # print('positionVecE1:',positionVecE1.shape)        #[61,5]
            # print('positionVecE1:',positionVecE1[position_e1[0]])
            # print('positionVecE1Dao:',positionVecE1Dao[position_e1[0]])
            # print('matrixW1PositionE1:',matrixW1PositionE1[j,0,0,0])
            # print('matrixW1PositionE1Dao:', matrixW1PositionE1Dao[j, 0, 0, 0])


        #原方法
        # for i in range(self.dimensionC):
        #     if math.fabs(grad[i])<1e-8:
        #         continue
        #     g2=g1[i]
        #     for j in range(self.window):
        #         if (tip[i]+j)>=length:
        #             continue
        #         position_v=sentence[tip[i]+j]
        #         position_e1=trainPositionE1[tip[i]+j]
        #         position_e2 = trainPositionE2[tip[i] + j]
        #         for k in range(self.dimension):
        #             matrixW1[j,k,0,i]-=g2*wordVecDao[position_v,k]
        #             wordVec[position_v,k]-=g2*matrixW1Dao[j,k,0,i]
        #         for k1 in range(self.dimensionWPE):
        #             matrixW1PositionE1[j,k1,0,i]-=g2*positionVecE1Dao[position_e1,k1]
        #             positionVecE1[position_e1, k1] -= g2 * matrixW1PositionE1Dao[j, k1, 0, i]
        #             matrixW1PositionE2[j, k1, 0, i] -= g2 * positionVecE2Dao[position_e2, k1]
        #             positionVecE2[position_e2, k1] -= g2 * matrixW1PositionE2Dao[j, k1, 0, i]


        #更新权重
        self.conv2d_v.set_weights([matrixW1,matrixB1])
        self.conv2d_p1.set_weights([matrixW1PositionE1,matrixW1PositionB1])
        self.conv2d_p2.set_weights([matrixW1PositionE2,matrixW1PositionB2])
        self.enc_v_embedding.set_weights([wordVec])
        self.enc_p1_embedding.set_weights([positionVecE1])
        self.enc_p2_embedding.set_weights([positionVecE2])


    #反向传播过程
    def backPropagation(self,label,bags_name,output,att_output,bagSize,rList,weight):
        #对matrixRelation(W)和matrixRelationPr(b)更新梯度
        g=(output*self.learning_rate).numpy()                    #[1,58]
        r1=label
        #正确的标签为负值
        g[0,r1]-=self.learning_rate                      #[1,58]

        #遍历所有关系
        # w=self.matrixRelation.get_weights()
        matrixRelation,matrixRelationPr=self.matrixRelation.get_weights()   #[230,58]  [58,]
        matrixRelationDao=matrixRelation.copy()

        g1=g*matrixRelationDao     #[230,58]
        g_MRW=np.dot(att_output.numpy().T,g)
        matrixRelation=matrixRelation-g_MRW                   #[230,58]
        #matrixRelation1=matrixRelation.copy()
        matrixRelationPr=matrixRelationPr-g.reshape((g.shape[1],))  #[58,]

        #下一层的梯度变化
        g1_temp=tf.reduce_sum(g1,axis=1)   #[230,]
        #保存梯度
        grad=np.zeros((bagSize,self.dimensionC),dtype=np.float32)  #[sen_n,230]

        #取出label关系的att_W
        att_W=self.att_W[r1].get_weights()[0]
        #att_W1=att_W.copy()
        # print('att_W:',att_W.shape)
        # print('att_W:', type(att_W))
        att_W_Dao=att_W.copy()
        #将梯度传递到att_W和matrixRelation

        #numpy实现
        grad+=np.dot(weight,tf.reshape(g1_temp,(1,self.dimensionC)))
        #grad_1=grad.copy()

        #tmp_sum=rList*weight
        tmp_sum=rList*weight      #[sen_n,230]
        tmp_sum=tf.reshape(tf.reduce_sum(tmp_sum,axis=0),(1,-1)).numpy()  #[1,230]

        tmp1=(tf.reshape(g1_temp,(1,-1))*rList*weight).numpy() #[sen_n,230]
        tmp2=np.dot(weight,tf.reshape(g1_temp,(1,-1))*tmp_sum)   #[sen_n,230]

        # tmp2_1=tf.reshape(matrixRelationDao[:,r1],(1,-1)) #[1,230]
        # tmp3=tmp1*tmp2_1    #[sen_n,230]
        # tmp4=np.dot(tmp3,att_W) #[sen_n,230]
        grad_temp1=np.dot(tmp1*tf.reshape(matrixRelationDao[:,r1],(1,-1)),att_W_Dao)   #[sen_n,230]
        #grad_temp2=grad_temp1*tmp_sum    #[sen_n,230]
        grad_temp2=np.dot(tmp2*tf.reshape(matrixRelationDao[:,r1],(1,-1)),att_W_Dao)   #[sen_n,230]
        grad += grad_temp1
        grad -= grad_temp2

        mR_grad1=tf.reduce_sum(np.dot(tmp1.T,rList)*att_W_Dao,axis=1).numpy()   #[230,]
        mR_grad2=tf.reduce_sum(np.dot(tmp2.T,rList)*att_W_Dao,axis=1).numpy()   #[230,]
        #mR_grad=np.dot(tmp1.T,rList)  #[230,230]
        # tmp5=mR_grad*att_W_Dao #[230,230]
        # tmp6=tf.reduce_sum(tmp5,axis=1)  #[230,]
        matrixRelation[:,r1]+=mR_grad1
        matrixRelation[:, r1] -=mR_grad2

        att_W_grad1=np.square(rList) #[sen_n,230]
        tmp7=(tf.reshape(g1_temp,(1,-1))*att_W_grad1*weight).numpy()  #[sen_n,230]
        #tmp8=tf.reduce_sum(tmp7,axis=0).numpy()   #[1,230]
        tmp9=(tf.reduce_sum(tmp7,axis=0)*matrixRelationDao[:,r1]).numpy()     #[1,230]
        tmp10=np.diag(tmp9)                      #[230,230]

        att_W_grad2=tmp2*rList   #[sen_n,230]
        #tmp11=tf.reduce_sum(att_W_grad2,axis=0).numpy()
        tmp12=(tf.reduce_sum(att_W_grad2,axis=0)*matrixRelationDao[:,r1]).numpy()
        tmp13=np.diag(tmp12)

        att_W+=tmp10
        att_W-=tmp13


        # #原方法
        # for i in range(self.dimensionC):
        #     g2=g1_temp[i]
        #     tmp_sum1=0
        #     for k in range(bagSize):
        #         # grad[k,i]+=g2*weight[k]
        #         for j in range(self.dimensionC):
        #             grad_1[k,j]+=g2*rList[k,i]*weight[k]*matrixRelationDao[i,r1]*att_W_Dao[j,i]
        #             matrixRelation1[i,r1]+=g2*rList[k,i]*weight[k]*rList[k][j]*att_W_Dao[j,i]
        #             if i==j:
        #                 att_W1[j,i] += g2 * rList[k,i] * weight[k] * rList[k,j] * matrixRelationDao[i,r1]
        #         tmp_sum1+=rList[k,i]*weight[k]
        #         #print('tmp_sum:',tmp_sum[0,i],'tmp_sum1:',tmp_sum1)
        #
        #     for k1 in range(bagSize):
        #         for j in range(self.dimensionC):
        #             grad_1[k1,j] -= g2 * tmp_sum1 * weight[k1] * matrixRelationDao[i,r1] * att_W_Dao[j,i]
        #             matrixRelation1[i,r1] -= g2 * tmp_sum1 * weight[k1] * rList[k1,j] * att_W_Dao[j,i]
        #             if i == j:
        #                 att_W1[j,i] -= g2 * tmp_sum1 * weight[k1] * rList[k1,j] * matrixRelationDao[i,r1]
        #
        # print('判断是否grad相同:',(grad.numpy()==grad_1).all())
        # print('判断是否mR相同:', (matrixRelation == matrixRelation1).all())
        # print('判断是否att_W相同：',(att_W==att_W1).all())

        # 将梯度应用
        self.matrixRelation.set_weights([matrixRelation, matrixRelationPr])
        self.att_W[r1].set_weights([att_W])

        for k in range(bagSize):
            i=self.data_loader.bagsTrain[bags_name][k]
            #print('i:',i)
            self.train_gradient(index=i,r=rList[k,:],tip=self.tip[k],grad=grad[k,:])


    def step(self,inputs,labels,training=True):
        with tf.GradientTape() as tape:
            y_pred,att_output,cnn_output,weight=self.call(inputs)  #[1,58],[1,230]
            #y_pred=y_pred.reshape((y_pred.shape[0],y_pred.shape[2]))  #[160,58]
            #print('y_pred:',y_pred)
            y=np.array(labels,dtype=np.int32)                #[1,]
            #y=y.reshape((y.shape[0],1))       #[1,1]
            loss=tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
            #grads=tape.gradient(loss,self.trainable_variables)

        # with open(logFile,'w+') as f:
        #     print('grads:',grads,file=f)
        if training:
            self.backPropagation(y,inputs,y_pred,att_output,len(self.data_loader.bagsTrain[inputs]),cnn_output,weight)
        #self.opt.apply_gradients(zip(grads,self.trainable_variables))
        #self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy(),y_pred


#data_loader = dataLoader()

def train():

    model = CNN_Selective_Attention(
        wordTotal=init.data_loader.wordTotal, PositionTotalE1=init.data_loader.PositionTotalE1,PositionTotalE2=init.data_loader.PositionTotalE2,
        dimension=init.data_loader.dimension, dimensionWPE=init.dimensionWPE, dimensionC=init.dimensionC,
        window=init.window, relationTotal=init.data_loader.relationTotal, data_loader=init.data_loader, dropoutRate=init.dropoutRate,
        learning_rate=init.learningRate,batchSize=init.batchSize,predictThreshold=init.predictThreshold)
    #模型热身
    model.set_flag(True)
    warm_x,warm_y=init.data_loader.get_train_batch(1)
    model.step(warm_x[0],warm_y[0],training=False)
    checkpoint=tf.train.Checkpoint(myModel=model)
    print('Warm up end.')

    #从最近训练的模型中恢复参数
    # try:
    #     model=keras.models.load_model(modelSavePath)
    # except:
    #     print('No saved model,create new...')
    #     data_loader = dataLoader()
    #     model = CNN_Selective_Attention(
    #         wordTotal=data_loader.wordTotal, PositionTotalE1=data_loader.PositionTotalE1,
    #         PositionTotalE2=data_loader.PositionTotalE2,
    #         dimension=data_loader.dimension, dimensionWPE=dimensionWPE, dimensionC=dimensionC,
    #         window=window, relationTotal=data_loader.relationTotal, data_loader=data_loader, dropoutRate=dropoutRate,
    #         learning_rate=learningRate,
    #         batchSize=batchSize)
    #     # 若为第一次训练需初始化模型参数
    #     model.set_flag(True)

    try:
        checkpoint.restore(tf.train.latest_checkpoint(modelDirectoryPath)).assert_consumed()
    except Exception as e:
        print(e)
        #若为第一次训练需初始化模型参数
        print('No saved model,create new...')
    manager=tf.train.CheckpointManager(checkpoint,directory=modelDirectoryPath,max_to_keep=3)

    # print('att_W_0:', model.att_W[0].get_weights())
    # print('att_W_1:', model.att_W[1].get_weights())
    # print('att_W_2:', model.att_W[2].get_weights())

    batch_num=len(init.data_loader.bagsTrain)//init.batchSize
    print('batch_num:',batch_num)

    #计时
    start = time.time()

    with open(trainLogFile,'w+') as f:
        for epoch in range(init.nPoch):
            for i in range(batch_num):
                batch_loss_list=np.zeros((init.batchSize,),dtype=np.float32)
                max_loss=0
                max_loss_traindata=''
                max_loss_predict=''
                train_x, train_y = init.data_loader.get_train_batch(init.batchSize)
                # print('train_x:',train_x)
                # print('train_y:',train_y)
                #print('train_y:',train_y.shape)
                for j in range(len(train_x)):
                    loss,y_pred=model.step(train_x[j],train_y[j])
                    batch_loss_list[j]=loss
                    if loss>max_loss:
                        max_loss=loss
                        max_loss_traindata=train_x[j]
                        max_loss_predict=init.data_loader.nam[np.argmax(y_pred)]

                batch_loss = np.mean(batch_loss_list)
                #每100个批次保存一次
                if i%20==0:
                    manager.save(checkpoint_number=(epoch+1)*i)
                    #model.save(modelSavePath)
                    print('epoch:',epoch,'batch:',i,'loss:',batch_loss)
                    print('epoch:', epoch, 'batch:', i, 'loss:', batch_loss,file=f)
                    print('max loss:', max_loss)
                    print('train data:', max_loss_traindata)
                    index=init.data_loader.bagsTrain[max_loss_traindata][0]
                    head=init.data_loader.headList[index]
                    tail=init.data_loader.tailList[index]
                    print('head:',head,'tail:',tail)
                    print('predict:', max_loss_predict)
                    print('Time taken for 20 batch {} sec\n'.format(time.time() - start))
                    start=time.time()

                #每500个批次测试一次
                if i%100==0 and i:
                    # 取出测试数据
                    test_bags, test_labels = init.data_loader.get_test_batch(1001)
                    test.test(model,test_bags,test_labels)



if __name__ == '__main__':
    train()