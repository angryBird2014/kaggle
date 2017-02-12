
import csv
import numpy as np
from sklearn import neighbors,tree,svm
def loadDataFromTrainCSV(filename):
    data = []
    label = []
    with open(filename) as csvfile:
        train_data = csv.reader(csvfile)
        next(train_data)
        for line in train_data:
            line += [line.pop(0)]
            line = list(map(int,line))
            data.append(line[:-1])
            label.append(line[-1])
    return preprocessData(np.array(data)),np.array(label)


def loadDataFromTestCSV(filename):
    data = []
    with open(filename) as csvfile:
        test_data = csv.reader(csvfile)
        next(test_data)
        for line in test_data:
            line += [line.pop(0)]
            line = list(map(int,line))
            data.append(line)
    return preprocessData(np.array(data))

def KNN(data,label,test_data):
    clf = neighbors.KNeighborsClassifier(n_jobs=3)
    clf.fit(data,label)
    print(clf)
    test_label = clf.predict(test_data)
    return test_label

def svmEstimulate(data,lable,test_data):
    svc = svm.SVC(kernel='rbf',C=8).fit(data,lable)
    test_label = svc.predict(test_data)
    return test_label

def decisionTree(data,label,test_data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(data,label)
    test_label = clf.predict(test_data)
    return test_label

def preprocessData(data):
    m,n = np.shape(data)
    tmp = data.reshape(m*n)
    for index,element in enumerate(tmp):
        if element != 0:
            tmp[index] = 1
    data = tmp.reshape(m,n)
    return data

def loadTrueLable(filename):
    label = []
    with open(filename) as file:
        TrueData = csv.reader(file)
        next(TrueData)
        for element in TrueData:
            label.append(int(element[1]))
    return label

def CaculateAccury(predict_lable,true_lable):
    count = 0
    for index in range(len(predict_lable)):
        if(predict_lable[index] == true_lable[index]):
            count+=1
    return float(count)/len(predict_lable)



if __name__ == '__main__':
    train_data,train_label = loadDataFromTrainCSV('train.csv')
    test_data = loadDataFromTestCSV('test.csv')
    #test_label = KNN(train_data,train_label,test_data)
    #test_label = decisionTree(train_data,train_label,test_data)
    test_label = svmEstimulate(train_data,train_label,test_data)
    print(test_label)
    true_lable = loadTrueLable('rf_benchmark.csv')
    print(CaculateAccury(test_label,true_lable))

    with open('predict_label.csv','w') as MyFile:
        MyWrite = csv.writer(MyFile)
        MyWrite.writerow(['ImageId','Label'])
        for i,index in enumerate(test_label):
            MyWrite.writerow([i+1,index])

