import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TaiTan():
    def readData(self,trainfilename,testfilename):
        x = pd.read_csv(trainfilename,lineterminator='\n',delimiter=',')

        x_2 = pd.read_csv(testfilename)
        y = pd.read_csv(testfilename)
        #print(x['Pclass'])
        Pclass1_female = np.mean(x.query('Pclass==1 and Sex == "female"')['Age'])
        Pclass1_male = np.mean(x.query('Pclass == 1 and Sex == "male"'))['Age']
        Pclass2_female = np.mean(x.query('Pclass == 2 and Sex == "female"'))['Age']
        Pclass2_male = np.mean(x.query('Pclass == 2 and Sex == "male"'))['Age']
        Pclass3_female = np.mean(x.query('Pclass == 3 and Sex == "female"'))['Age']
        Pclass3_male = np.mean(x.query('Pclass == 3 and Sex == "male"'))['Age']

        print(Pclass1_female,Pclass1_male,Pclass2_female,Pclass2_male,Pclass3_female,Pclass3_male)

        x.query('Pclass==1 and Sex == "female"')['Age'].fillna(Pclass1_female)
        x.query('Pclass==1 and Sex == "male"')['Age'].fillna(Pclass1_male)
        x.query('Pclass==2 and Sex == "female"')['Age'].fillna(Pclass2_female)
        x.query('Pclass==2 and Sex == "male"')['Age'].fillna(Pclass2_male)
        x.query('Pclass==3 and Sex == "female"')['Age'].fillna(Pclass3_female)
        x.query('Pclass==3 and Sex == "male"')['Age'].fillna(Pclass3_male)

        return x,y,x_2

if __name__ == '__main__':
    trainfilename = 'train.csv'
    testfilename = 'test.csv'
    T = TaiTan()
    T.readData(trainfilename,testfilename)
