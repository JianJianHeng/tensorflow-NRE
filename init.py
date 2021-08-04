#初始化数据
import random
import numpy as np

binFilePath='data/vec.bin'
vecFilePath='data/RE/word2vec1_01.txt'
REFilePath='data/RE/relation2id.txt'
trainFilePath='data/RE/train.txt'
testFilePath='data/RE/test.txt'

#embedding
emb_word_dim=50
dimensionWPE=5    #position embedding大小
#CNN encoder
dimensionC=230    #卷积核
window=3          #滑动窗口大小

#train
dropoutRate=0.5
batchSize=50
learningRate=0.001
NA_relation_proportion=0.93   #训练批次中NA关系的比例
nPoch=5
#test
predictThreshold=0.2    #当预测的概率大于该阈值时判断存在关系

TEST_NA_relation_proportion=0.98

class dataLoader():
    def __init__(self,train_NA_proportion,test_Na_proportion):
        self.NA_relation_proportion=train_NA_proportion
        self.TEST_NA_relation_proportion=test_Na_proportion

        self.limit=30      #word与head和tail的最大距离限制
        #表示与E1实体和E2实体的最大距离和最小距离
        self.PositionMinE1=0
        self.PositionMaxE1=0
        self.PositionTotalE1=0
        self.PositionMinE2=0
        self.PositionMaxE2=0
        self.PositionTotalE2=0

        #读取word2vec信息
        with open(binFilePath,'rb') as f:
            context=f.readline().decode('utf-8').split(' ')
        self.wordTotal=int(context[0])
        self.wordTotal+=1
        self.dimension=int(context[1])
        print("wordTotal:\t",self.wordTotal,"\tdimension:\t",self.dimension)

        self.wordVec=np.zeros((self.wordTotal,self.dimension),dtype=np.float32)   #word2Vec矩阵
        self.wordList={}                                                        #word字典
        self.wordMapping={}
        with open(vecFilePath,'r',encoding='utf-8') as f:
            wvec=f.readline().split(' ')
            while wvec[0]:
                for i in range(self.dimension):
                    self.wordVec[len(self.wordList)][i]=float(wvec[i+1])
                self.wordMapping[wvec[0]]=len(self.wordList)
                self.wordList[len(self.wordList)]=wvec[0]
                wvec=f.readline().split(' ')

        #读取关系数据
        self.relationMapping = {}
        self.nam={}
        with open(REFilePath,'r') as f:
            relation=f.readline().split(' ')
            while relation[0]:
                self.relationMapping[relation[0]]=int(relation[1])
                self.nam[int(relation[1])] = relation[0]
                relation=f.readline().split(' ')
        # for rel,id in self.relationMapping.items():
        #     print(rel)
        self.relationTotal=len(self.nam)

        #读取训练数据
        self.bagsTrain={}     #多示例训练数据集
        self.bagsTrain_NA={}  #NA关系训练数据集
        self.bagsTrain_notNA={}  #非NA关系训练数据集
        self.headList={}      #e1列表
        self.tailList={}      #e2列表
        self.relationList={}  #关系列表(标签)
        self.trainLength={}   #记录训练的句子长度

        # 计算每个word与head和tail的距离
        self.trainList = {}
        self.trainPositionE1 = {}
        self.trainPositionE2 = {}
        count=0               #记录有多少个训练数据
        with open(trainFilePath,'r') as f:
            trainData=f.readline().split('\t')
            while trainData[0]:
                #读取实体数据
                head_s=trainData[2]
                if head_s in self.wordMapping.keys():
                    head = self.wordMapping[head_s]
                else:  # 若不存在词典中，标记为UNK
                    head = self.wordMapping['UNK']
                tail_s=trainData[3]
                if tail_s in self.wordMapping.keys():
                    tail = self.wordMapping[tail_s]
                else:
                    tail = self.wordMapping['UNK']
                rel=trainData[4]
                sen=trainData[5].split(' ')
                #构造训练数据
                if head_s+'\t'+tail_s+'\t'+rel in self.bagsTrain.keys():
                    self.bagsTrain[head_s+'\t'+tail_s+'\t'+rel].append(count)
                else:
                    self.bagsTrain[head_s+'\t'+tail_s+'\t'+rel]=[]
                    self.bagsTrain[head_s + '\t' + tail_s + '\t' + rel].append(count)


                #读取句子
                leng=0         #句子长度
                lefnum=0       #head的位置
                rignum=0       #tail的位置
                tmpp={}        #将word转换成id后连接成序列
                for word in sen:
                    if word is '###END###':
                        break
                    if word in self.wordMapping.keys():
                        gg=self.wordMapping[word]
                    else:
                        gg=self.wordMapping['UNK']
                    if gg==head:
                        lefnum=leng
                    if gg==tail:
                        rignum=leng
                    tmpp[leng]=gg
                    leng += 1

                #将句子中的信息加入训练数据中
                try:
                    self.relationList[count]=self.relationMapping[rel]
                except:
                    #将文本未收集到的关系加入relationMapping中
                    self.relationMapping[rel]=self.relationTotal
                    self.nam[self.relationTotal] = rel
                    self.relationTotal += 1
                    self.relationList[count] = self.relationMapping[rel]

                self.headList[count]=head
                self.tailList[count]=tail
                self.trainLength[count]=leng

                #构造NA和notNA的数据集
                if self.relationList[count]==0:
                    if head_s + '\t' + tail_s + '\t' + rel in self.bagsTrain_NA.keys():
                        self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel].append(count)
                    else:
                        self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel] = []
                        self.bagsTrain_NA[head_s + '\t' + tail_s + '\t' + rel].append(count)
                else:
                    if head_s + '\t' + tail_s + '\t' + rel in self.bagsTrain_notNA.keys():
                        self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel].append(count)
                    else:
                        self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel] = []
                        self.bagsTrain_notNA[head_s + '\t' + tail_s + '\t' + rel].append(count)

                con = []      #con是训练数据
                conl = []   #conl是与head的距离
                conr = []   #conr是与tail的距离
                for i in range(leng):
                    con.append(tmpp[i])
                    conl.append(lefnum - i)
                    conr.append(rignum - i)
                    if conl[i] >= self.limit:
                        conl[i] = self.limit
                    if (conr[i] >= self.limit):
                        conr[i] = self.limit
                    if (conl[i] <= -self.limit):
                        conl[i] = -self.limit
                    if (conr[i] <= -self.limit):
                        conr[i] = -self.limit
                    if (conl[i] > self.PositionMaxE1):
                        self.PositionMaxE1 = conl[i]
                    if (conr[i] > self.PositionMaxE2):
                        self.PositionMaxE2 = conr[i]
                    if (conl[i] < self.PositionMinE1):
                        self.PositionMinE1 = conl[i]
                    if (conr[i] < self.PositionMinE2):
                        self.PositionMinE2 = conr[i]

                self.trainList[count]=con
                self.trainPositionE1[count]=conl
                self.trainPositionE2[count]=conr

                count+=1
                trainData = f.readline().split('\t')

        # 读取测试数据
        self.bagsTest = {}  # 测试数据集
        #self.bagsTest_notNA={}  #非NA的数据集
        self.tot=0   #非NA关系的bags数，用于召回率计算使用
        self.bagsTest_Label = {}  # 最终所有bags正确的列表

        self.testHeadList = {}  # e1列表
        self.testTailList = {}  # e2列表
        self.testRelationList = {}  # 关系列表(标签)
        self.testTrainLength = {}  # 记录测试的句子长度

        # 计算每个word与head和tail的距离
        self.testTrainList = {}
        self.testPositionE1 = {}
        self.testPositionE2 = {}
        count = 0  # 记录有多少个测试数据
        with open(testFilePath, 'r') as f:
            trainData = f.readline().split('\t')
            while trainData[0]:
                # 读取实体数据
                head_s = trainData[2]
                if head_s in self.wordMapping.keys():
                    head = self.wordMapping[head_s]
                else:  # 若不存在词典中，标记为UNK
                    head = self.wordMapping['UNK']
                tail_s = trainData[3]
                if tail_s in self.wordMapping.keys():
                    tail = self.wordMapping[tail_s]
                else:
                    tail = self.wordMapping['UNK']
                rel = trainData[4]
                sen = trainData[5].split(' ')
                #用于预测的测试包数据中字典键不包含关系rel
                if head_s+'\t'+tail_s in self.bagsTest.keys():
                    self.bagsTest[head_s+'\t'+tail_s].append(count)
                else:
                    self.bagsTest[head_s+'\t'+tail_s]=[]
                    self.bagsTest[head_s + '\t' + tail_s].append(count)

                # 读取句子
                leng = 0  # 句子长度
                lefnum = 0  # head的位置
                rignum = 0  # tail的位置
                tmpp = {}  # 将word转换成id后连接成序列
                for word in sen:
                    if word is '###END###':
                        break
                    if word in self.wordMapping.keys():
                        gg = self.wordMapping[word]
                    else:
                        gg = self.wordMapping['UNK']
                    if gg == head:
                        lefnum = leng
                    if gg == tail:
                        rignum = leng
                    tmpp[leng] = gg
                    leng += 1

                # 将句子中的信息加入测试数据中
                self.testRelationList[count] = self.relationMapping[rel]
                try:
                    self.testRelationList[count] = self.relationMapping[rel]
                except:
                    # 将文本未收集到的关系加入relationMapping中
                    self.relationMapping[rel] = self.relationTotal
                    self.nam[self.relationTotal] = rel
                    self.relationTotal += 1
                    self.testRelationList[count] = self.relationMapping[rel]
                #构建最终的label数据
                if head_s+'\t'+tail_s+'\t'+rel in self.bagsTest_Label.keys():
                    self.bagsTest_Label[head_s+'\t'+tail_s+'\t'+rel].append(count)
                else:
                    self.bagsTest_Label[head_s+'\t'+tail_s+'\t'+rel]=[]
                    self.bagsTest_Label[head_s + '\t' + tail_s+'\t'+rel].append(count)
                    # 统计正样本
                    if self.relationMapping[rel] != 0:
                        self.tot += 1

                self.testHeadList[count] = head
                self.testTailList[count] = tail
                self.testTrainLength[count] = leng

                con = []  # con是测试数据
                conl = []  # conl是与head的距离
                conr = []  # conr是与tail的距离
                for i in range(leng):
                    con.append(tmpp[i])
                    conl.append(lefnum - i)
                    conr.append(rignum - i)
                    if conl[i] >= self.limit:
                        conl[i] = self.limit
                    if (conr[i] >= self.limit):
                        conr[i] = self.limit
                    if (conl[i] <= -self.limit):
                        conl[i] = -self.limit
                    if (conr[i] <= -self.limit):
                        conr[i] = -self.limit
                    if (conl[i] > self.PositionMaxE1):
                        self.PositionMaxE1 = conl[i]
                    if (conr[i] > self.PositionMaxE2):
                        self.PositionMaxE2 = conr[i]
                    if (conl[i] < self.PositionMinE1):
                        self.PositionMinE1 = conl[i]
                    if (conr[i] < self.PositionMinE2):
                        self.PositionMinE2 = conr[i]

                self.testTrainList[count] = con
                self.testPositionE1[count] = conl
                self.testPositionE2[count] = conr

                count += 1
                trainData = f.readline().split('\t')

        self.bagsTest_notNA = {}  # 测试包中存在非NA关系的bags
        self.bagsTest_NA = {}  # 测试包中关系全为NA的bags
        for bagsName,testList in self.bagsTest.items():
            #若包中存在不为NA关系的元素
            list=[]
            for index in testList:
                list.append(self.testRelationList[index])
            # print('testList:',testList)
            # print('list:',list)
            if np.all(np.array(list)==0):
                self.bagsTest_NA[bagsName]=testList
            else:
                self.bagsTest_notNA[bagsName]=testList

        #处理训练和测试的PositionE,将距离全部变成正数
        for i in range(len(self.trainPositionE1)):
            leng=self.trainLength[i]
            for j in range(leng):
                self.trainPositionE1[i][j]=self.trainPositionE1[i][j]-self.PositionMinE1
            for j in range(leng):
                self.trainPositionE2[i][j]=self.trainPositionE2[i][j]-self.PositionMinE2

        for i in range(len(self.testPositionE1)):
            leng=self.testTrainLength[i]
            for j in range(leng):
                self.testPositionE1[i][j]=self.testPositionE1[i][j]-self.PositionMinE1
            for j in range(leng):
                self.testPositionE2[i][j]=self.testPositionE2[i][j]-self.PositionMinE2

        print("trainList:",len(self.trainList))
        print("bagsTrain:", len(self.bagsTrain))
        print('bagsTrain_NA:',len(self.bagsTrain_NA))
        print('bagsTrain_notNA:',len(self.bagsTrain_notNA))

        print("testTrainList:", len(self.testTrainList))
        print('bagsTest:', len(self.bagsTest))
        print('bagsTest_Label:', len(self.bagsTest_Label))
        print('tot:', self.tot)
        print('bagsTest_NA:',len(self.bagsTest_NA))
        print('bagsTest_notNA:', len(self.bagsTest_notNA))

        print("relationTotal:\t", self.relationTotal)
        print(self.PositionMinE1,' ',self.PositionMaxE1,' ',self.PositionMinE2,' ',self.PositionMaxE2)
        self.PositionTotalE1=self.PositionMaxE1-self.PositionMinE1+1
        self.PositionTotalE2 = self.PositionMaxE2 - self.PositionMinE2 + 1

        print("Init end...")

    def get_train_batch(self,n):
        bagsBatch=[]
        relationBatch=[]
        #数据集中NA关系和not_NA关系的个数
        NA_num=int(n*self.NA_relation_proportion)
        not_NA_num=n-NA_num
        #添加NA_num个NA关系和not_NA_num个非NA关系的数据
        for i in range(NA_num):
            bx=random.choice(list(self.bagsTrain_NA))
            bagsBatch.append(bx)
            relationBatch.append(0)
        for i in range(not_NA_num):
            bx = random.choice(list(self.bagsTrain_notNA))
            by = self.relationList[self.bagsTrain[bx][0]]
            bagsBatch.append(bx)
            relationBatch.append(by)

        return bagsBatch,relationBatch

    #批量获取测试数据
    def get_test_batch(self, n):
        testBags=[]
        testLabels=[]
        # 测试集中NA关系和not_NA关系的个数
        NA_num = int(n * self.TEST_NA_relation_proportion)
        not_NA_num = n - NA_num
        testBags_NA=random.sample(list(self.bagsTest_NA),NA_num)
        # print('testBags1:',testBags)
        # print('testBags1:', type(testBags))
        for bx in testBags_NA:
            testBags.append(bx)
            by=set()
            for index in self.bagsTest[bx]:
                by.add(self.testRelationList[index])
            testLabels.append(by)
        testBags_notNA = random.sample(list(self.bagsTest_notNA), not_NA_num)
        for bx in testBags_notNA:
            testBags.append(bx)
            by=set()
            for index in self.bagsTest[bx]:
                by.add(self.testRelationList[index])
            testLabels.append(by)

        test_data=zip(testBags,testLabels)
        test_data = list(test_data)
        random.shuffle(test_data)
        # print('test_data:',test_data)
        test_bags = []
        test_labels = []
        test_bags[:], test_labels[:] = zip(*test_data)
        return test_bags,test_labels

    #将所有测试数据打包返回
    def get_test_all_batch(self):
        testBags = list(self.bagsTest.keys())
        testLabels = []
        #print('testBags:',len(testBags))
        #print('testBags:', type(testBags))
        for bx in testBags:
            by=set()
            for index in self.bagsTest[bx]:
                by.add(self.testRelationList[index])
            testLabels.append(by)

        test_data = zip(testBags, testLabels)
        test_data = list(test_data)
        random.shuffle(test_data)
        test_bags = []
        test_labels = []
        test_bags[:], test_labels[:] = zip(*test_data)
        return test_bags, test_labels

data_loader = dataLoader(NA_relation_proportion,TEST_NA_relation_proportion)

def att_W_init(shape, dtype=np.float32):
    return np.identity(shape[0])

def enc_V_init(shape,dtype=np.float32):
    return data_loader.wordVec