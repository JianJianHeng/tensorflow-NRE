#绘制结果图
import matplotlib.pyplot as plt

resultFilePath='test_results/7_25所有数据测试结果.txt'

batchs=[]
acc=[]
precision=[]
recall=[]

#读取结果
with open(resultFilePath,'r',encoding='utf-8') as f:
    allResults=f.readline().split('\n')[0].split(' ')
    #print(allResults)
    while(allResults[0]):
        batchs.append(int(allResults[2]))
        acc.append(float(allResults[4]))
        precision.append(float(allResults[6]))
        recall.append(float(allResults[8]))
        allResults = f.readline().split('\n')[0].split(' ')

#绘制precision和recall图
plt.plot(recall,precision,'r',label='Testing recall&acc')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

#绘制acc图
plt.clf() #清除数字
plt.plot(batchs,acc,'r',label='Testing acc')
plt.title('Training and validation accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()