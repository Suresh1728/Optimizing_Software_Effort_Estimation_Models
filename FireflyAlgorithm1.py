import numpy as np
import random as rand
import math
import time
import pandas as pd
class FireflyAlgorithm:

    def __init__(self, D, Lb, Ub, n, alpha, beta0, gamma, theta, iter_max,size,ae,func):
        self.D = D
        self.Lb = Lb
        self.Ub = Ub
        self.n = n
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.theta = theta
        self.iter_max = iter_max
        self.func = func
        self.populationArray = np.zeros((n, D))
        self.functionArray = np.zeros(n)
        self.tmpArray = np.zeros(D)
        self.size=size
        self.ae=ae
    
    def init_FA(self):
        for i in range(self.n):
            for j in range(self.D):
                self.populationArray[i][j] = rand.uniform(self.Lb, self.Ub)
            self.functionArray[i] = self.func(self.populationArray[i,:], self.D,self.size,self.ae)
                
    def update(self, i, j):
        scale = self.Ub - self.Lb
        r = 0
        for k in range(self.D):
            r += (self.populationArray[i][k] - self.populationArray[j][k])**2
        beta = self.beta0*math.exp(-self.gamma*r)
        for k in range(self.D):
            steps = (self.alpha*self.theta)*(rand.random() - 0.5)*scale
            self.tmpArray[k] = self.populationArray[i][k] + beta*(self.populationArray[j][k] - self.populationArray[i][k]) + steps
        if(self.func(self.tmpArray,self.D,self.size,self.ae) < self.functionArray[i]):
            for k in range(self.D):
                self.populationArray[i][k] = self.tmpArray[k]
            self.functionArray[i] = self.func(self.tmpArray, self.D,self.size,self.ae)
            
    def doRun(self):
        start = time.time()
        self.init_FA()
        for gen in range(self.iter_max):
            print("Generation ", gen+1)
            for i in range(self.n):
                for j in range(self.n):
                    if(self.functionArray[i] > self.functionArray[j] and i != j):
                        self.update(i,j)
            print(self.populationArray)
            print(self.functionArray)
        end = time.time()
        #print("：%f " % (end - start))
        return self.functionArray.min()


def CostEstimation(x,D,size,ae):
    #print(x)
    mre=[]
    pe=[]
    rsme=0
    for i in range(0,len(size)):
       pe.append(x[0]*(size[i])**x[1])
       mre.append((ae[i]-pe[i])**2)
    y=sum(mre)
    rsme=math.sqrt(y/len(size))
    return rsme






file_path = "C:/Users/ketha/Desktop/Projects/major/code/dataset/albrecht2.csv"
df=pd.read_csv(file_path)
size=df["size"]
ae=df["Effort"]
#FireflyAlgorithm(D, Lb, Ub, n, alpha, beta0, gamma, theta, iter_max, func)
FA = FireflyAlgorithm(2, -5, 5, 63, 0.1, 1.0, 0.01, 0.97, 10,size,ae,CostEstimation)
ans = FA.doRun()
print("Minimal",ans)

