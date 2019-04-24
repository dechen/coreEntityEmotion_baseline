#coding:utf-8

class LSTM():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    #推论
    def predict(self, test_data=None):
        ###如果为指定测试数据集，则使用模型测试数据集
        if not test_data and self.test_data:
            test_data = self.test_data
        else:
            print("请输入测试数据集！")

            ###测试代码

    #训练
    def train(self, train_data=None):

        if not train_data and self.train_data:
            train_data = self.train_data
        else:
            print("请输入训练数据集！")

        pass


    #数据预处理，将输入数据整理成当前模型的输入格式
    def preprocess(self):
        pass

if __name__ == "__main__":
    train_data = None
    test_data = None
    lstm = LSTM(train_data, test_data)
