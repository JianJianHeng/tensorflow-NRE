import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import sys
import init


class PCNN_ATT(keras.Model):
    def __init__(self,dimension,dimensionWPE,dimensionC,window,
                 data_loader,dropoutRate,learning_rate,batchSize,predictThreshold=0.75):
        super().__init__()
        self.dimension=dimension
        self.dimensionC=dimensionC
        self.dimensionWPE=dimensionWPE
        self.window=window
        self.data_loader=data_loader
        self.learning_rate=learning_rate
        self.predictThreshold=predictThreshold

        self.con=math.sqrt(6.0/(dimensionC+self.data_loader.relationTotal))
        self.con1=math.sqrt(6.0/((dimensionWPE+dimension)*window))

        #tip列表(此处初始化无作用)
        self.tip=np.zeros((1,self.dimensionC),dtype=np.int32)

        #手动设置输入
        #self._set_inputs(tf.TensorSpec([None,self.dimension],tf.float32,name='inputs'))
        #embedding
        self.enc_v_embedding=keras.layers.Embedding(
            input_dim=self.data_loader.wordTotal,output_dim=dimension,embeddings_initializer=init.enc_V_init
        )  #输入[seq_len] 输出[seq_len,dimension]
        self.enc_p1_embedding=keras.layers.Embedding(
            input_dim=self.data_loader.PositionTotalE1,output_dim=dimensionWPE,embeddings_initializer=tf.initializers.RandomNormal(-self.con1,self.con1)
        )  #输入[seq_len] 输出[seq_len,dimensionWPE]
        self.enc_p2_embedding = keras.layers.Embedding(
            input_dim=self.data_loader.PositionTotalE2, output_dim=dimensionWPE,
            embeddings_initializer=tf.initializers.RandomNormal(-self.con1, self.con1)
        )  # 输入[seq_len] 输出[seq_len,dimensionWPE]

        #encoder层
        self.conv1d=keras.layers.Conv1D(dimensionC,window,padding='same',) #input:[seq_len,dimension] output:[seq_len-windows+1,dimensionC]
        self.maxPooling=keras.layers.GlobalMaxPool1D()

        #selective_attention
        #query vector
        self.matrixRelation=keras.layers.Dense(units=self.data_loader.relationTotal,kernel_initializer='RandomNormal')

        #dropout
        self.dropout=keras.layers.Dropout(dropoutRate,input_shape=(dimensionC,))

        #output
        self.testOutput=keras.layers.Dense(units=1)

        #optimizer
        self.opt=keras.optimizers.SGD(learning_rate=self.learning_rate)

    #获得三段的embedding
    def get_segement(self,emb,e1,e2):
        #判断两个实体的相对位置
        if e1<e2:
            head=e1
            tail=e2
        else:
            head=e2
            tail=e1

        left=emb[:,:head+1]
        mid=emb[:,head:tail+1]
        right=emb[:,tail:]

        return left,mid,right

    def Piece_Wise_CNN(self, sen, p1, p2,positionE1,positionE2, index, training=True):
        embedded_v = self.enc_v_embedding(sen)  # [1,seq_len,dimension]
        embedded_p1 = self.enc_p1_embedding(p1)  # [1,seq_len,dimensionWPE]
        embedded_p2 = self.enc_p2_embedding(p2)  # [1,seq_len,dimensionWPE]

        #将三个embedd信息拼在一块
        embedding=tf.concat([embedded_v,embedded_p1,embedded_p2],axis=2)  #[1,se1_len,dimesion+2*dimensionWPE]
        #获得三段信息
        left,mid,right=self.get_segement(embedding,positionE1,positionE2)

        #CNN
        cnn_output_seg1=self.conv1d(left) #[1,1eft_len,dimension]
        cnn_output_seg2=self.conv1d(mid)
        cnn_output_seg3=self.conv1d(right)

        #池化
        max_pool_seg1=self.maxPooling(cnn_output_seg1)  #[1,dimensionC]
        max_pool_seg2=self.maxPooling(cnn_output_seg2)  #[1,dimensionC]
        max_pool_seg3=self.maxPooling(cnn_output_seg3)  #[1,dimensionC]

        out=tf.concat([max_pool_seg1,max_pool_seg2,max_pool_seg3],1) #[1,dimensionC*3]

        return out

    def train_batch(self, inputs):
        bagsName = inputs
        bagsSize = len(self.data_loader.bagsTrain[bagsName])

        rList = np.zeros((bagsSize, self.dimensionC*3), dtype=np.float32)
        rel = self.data_loader.relationList[self.data_loader.bagsTrain[bagsName][0]]

        for i in range(bagsSize):
            index = self.data_loader.bagsTrain[bagsName][i]
            rel1 = self.data_loader.relationList[index]
            if rel1 != rel:
                print('数据处理错误...')
                sys.exit(1)
            sen=tf.expand_dims(np.array(self.data_loader.trainList[index]),axis=0)
            p1=tf.expand_dims(np.array(self.data_loader.trainPositionE1[index]),axis=0)
            p2=tf.expand_dims(np.array(self.data_loader.trainPositionE2[index]),axis=0)
            positionE1=np.where(np.array(self.data_loader.trainPositionE1[index])==(-self.data_loader.PositionMinE1))  #e1位置
            positionE2=np.where(np.array(self.data_loader.trainPositionE2[index])==(-self.data_loader.PositionMinE2))  #e2位置

            rList[i, :] = self.Piece_Wise_CNN(sen,p1,p2,positionE1[0][0],positionE2[0][0],i)

        logits=self.testOutput(rList)

        return logits

    def backPropagation(self,loss,labels):
        with tf.GradientTape() as tape:
            grads1 = tape.gradient(loss, [self.variables[-2],self.variables[-1]])
            self.opt.apply_gradients(grads_and_vars=zip(grads1, [self.variables[-2],self.variables[-1]]))
            croospy2=tf.convert_to_tensor(np.ones((labels.shape[0],self.dimensionC*3)))
            grads2=tape.gradient(croospy2,self.conv1d.variables)
            self.opt.apply_gradients(grads_and_vars=zip(grads2,self.conv1d.trainable_variables))

    def step(self,inputs,labels,training=True):
        with tf.GradientTape() as tape:
            y_pred=self.train_batch(inputs)  #[1,58],[1,230]
            y=np.zeros(len(self.data_loader.bagsTrain[inputs]))
            loss=tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred)
            print('loss:',loss)
            print('loss:',loss.shape)
            print('loss:',type(loss))

            #loss=tf.convert_to_tensor(4.5)
        #grads = type.gradient(loss, self.variables)
        #print('variables:',self.variables)
        if training:
            self.backPropagation(loss,labels)


def train():
    model = PCNN_ATT(
        dimension=init.data_loader.dimension, dimensionWPE=init.dimensionWPE, dimensionC=init.dimensionC,
        window=init.window, data_loader=init.data_loader, dropoutRate=init.dropoutRate,
        learning_rate=init.learningRate, batchSize=init.batchSize, predictThreshold=init.predictThreshold)

    train_x, train_y = init.data_loader.get_train_batch(1)
    model.step(train_x[0],train_y[0])

if __name__ == '__main__':
    train()