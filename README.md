tensorflow-NRE
====
本项目目前只实现了论文中的CNN+ATT方法，PCNN+ATT方法正在复现中。且由于搭建模型后不能自动传导梯度，反向传播算法是参考源代码手动实现的，训练过程比较慢。
## Background
使用Tensorflow2复现2016年ACL中的经典论文*Neural Relation Extraction with Selective Attention over Instances*  
[论文源代码地址](https://github.com/thunlp/NRE)  

## Environment
` OS: WIN 10 `  
` Python: 3.7 `  
` Tensorflow-gpu: 2.2.0 `  
` CUDA: 10.2 `  
` GPU: GTX 1050 `  
  
## Usage
` init.py 处理数据 `  
` CNN_ATT_train.py 训练CNN+ATT模型 `  
` CNN_ATT_test.py 测试CNN+ATT模型的效果 `  
` ResultPolt 绘制PR图 `
  
## Result
复现效果PR图:  
![](https://github.com/JianJianHeng/tensorflow-NRE/raw/main/png/PrecisionRecallGraph.png)
ACC图：
![](https://github.com/JianJianHeng/tensorflow-NRE/raw/main/png/AccGraph.png)
  
*ps:* 以下为取得上图效果中的模型文件，下载后将文件全部解压至model文件夹即可。  
链接：https://pan.baidu.com/s/1c260BA_BsHz-EL9B5_NaCA 
提取码：p0k6