
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
from matplotlib import pyplot as plt

df_og = pd.read_excel('putanja2.xlsx')
df=df_og

X_D=df['X_D']
Y_D=df['Y_D']
X_L=df['X_L']
Y_L=df['Y_L']

offset=[]
for i in range(len(df)):
    offset.append(.5)

def calc_K(offset):
    X_K=[]
    Y_K=[]
    for i in range(len(df)):
        x_k=X_D[i]*offset[i]+X_L[i]*(1-offset[i])
        y_k=Y_D[i]*offset[i]+Y_L[i]*(1-offset[i])
        X_K.append(x_k)
        Y_K.append(y_k)
    return(X_K,Y_K)
def krivina(x1,y1,x2,y2,x3,y3):
    x, y, z = complex(x1,y1),complex(x2,y2),complex(x3,y3)
    w = z-x
    w /= y-x
    try:
        c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
        return (1/abs(c+x))
    except:
        return (0)
def eval_K(X_K,Y_K):
    eval_l=[]
    for i in range(1,len(df)-1):
        x1=X_K[i-1]
        x2=X_K[i]
        x3=X_K[i+1]
        y1=Y_K[i-1]
        y2=Y_K[i]
        y3=Y_K[i+1]
        kr=krivina(x1,y1,x2,y2,x3,y3)
        eval_l.append(kr)
    return(eval_l,sum(eval_l))

X_K,Y_K=calc_K(offset)
K_lista,K=eval_K(calc_K(offset)[0],calc_K(offset)[1])

plt.plot(K_lista)
plt.show()


def f(X):
    ret = eval_K(calc_K(X)[0],calc_K(X)[1])[1]
    return ret**3

varbound=np.array([[0,1]]*len(offset))

algorithm_param = {'max_num_iteration': 500,\
                   'population_size':50000,\
                   'mutation_probability':0.9,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.6,\
                   'parents_portion': 0.05,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':10}

model=ga(function=f,
         dimension=len(df),
         variable_type='real',
         variable_boundaries=varbound,
         algorithm_parameters=algorithm_param)

model.run()


calc_offset=model.output_dict['variable']
X_K_calc,Y_K_calc=calc_K(calc_offset)

plt.plot(df['X_D'],df['Y_D'],color='green')
plt.plot(df['X_L'],df['Y_L'],color='green')
plt.plot(X_K_calc,Y_K_calc,color='orange')


#help=[0,.2,.4,.6,.8,.85,.8,.6,.4,.2,0] ima eval 10 na -6 / to je otp prava 


