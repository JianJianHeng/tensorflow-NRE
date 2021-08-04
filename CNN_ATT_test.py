#测试模型的效果
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import CNN_ATT_train as train
import init

testBatch=1701

def test(model,test_bags,test_labels):
    precision = 0.0
    accuracy = 0.0
    total = 0
    recall = 0.0
    right = 0
    TP = 0
    FP = 0
    for i in range(len(test_bags)):
        bagsName = test_bags[i]
        predict = model.inference(bagsName)
        # print(predict)
        for pred in predict:
            if pred in test_labels[i]:  # 预测正确
                right += 1
                if pred != 0:
                    TP += 1
            else:  # 预测错误
                if pred != 0:
                    FP += 1

        total += len(test_labels[i])
        if i % 100 == 0 and i:
            try:
                precision=TP/(TP+FP)
            except ZeroDivisionError:
                precision=0
            print('bags Sum:', i, 'accuracy:', right / total, 'precision:', precision, 'recall:',
                  TP / init.data_loader.tot)

def test_main():

    model = train.CNN_Selective_Attention(
        wordTotal=init.data_loader.wordTotal, PositionTotalE1=init.data_loader.PositionTotalE1,
        PositionTotalE2=init.data_loader.PositionTotalE2,
        dimension=init.data_loader.dimension, dimensionWPE=init.dimensionWPE, dimensionC=init.dimensionC,
        window=init.window, relationTotal=init.data_loader.relationTotal, data_loader=init.data_loader, dropoutRate=init.dropoutRate,
        learning_rate=init.learningRate,
        batchSize=init.batchSize)
    # 模型热身
    model.set_flag(True)
    warm_x, warm_y = init.data_loader.get_train_batch(1)
    model.step(warm_x[0], warm_y[0],training=False)
    checkpoint = tf.train.Checkpoint(myModel=model)
    print('Warm up end.')
    try:
        checkpoint.restore(tf.train.latest_checkpoint(train.modelDirectoryPath)).assert_consumed()
    except Exception as e:
        print(e)
        #若为第一次训练需初始化模型参数
        print('No saved model,exit...')
        exit(1)

    #取出测试数据
    test_bags,test_labels=init.data_loader.get_test_batch(testBatch)
    #test_bags, test_labels = init.data_loader.get_test_all_batch()

    test(model,test_bags,test_labels)




if __name__ == '__main__':
    test_main()