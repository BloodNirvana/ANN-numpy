import numpy as np
import ANN

output_layer_unit_amount=1   #1 unit, usually for 0 or 1 classification
hidden_layer2_unit_amount=10 
hidden_layer1_unit_amount=10 
train_iteration=2000
learn_rate=0.01

layer_structure=[hidden_layer1_unit_amount, hidden_layer2_unit_amount, output_layer_unit_amount] #freely expand

input_data_dimensions=4 #a train sample like [1,2,3,4]
w=ANN.initialize_weights(layer_structure, input_data_dimensions)

#data example 
# x=[[1,3,4,5],[1,3,4,5],[1,3,4,5],[1,3,4,5],[1,3,4,5],....]
# y=[0,1,0,1,1,.....]

a=list(range(-5,5))
b=list(range(-5,5))
c=list(range(-5,5))
d=list(range(-5,5))

x=[] #train samples
y=[] #labels
for a_ in a:
    for b_ in b:
        for c_ in c:
            for d_ in d:

                x.append([a_,b_,c_,d_])
                y_=2*a_+3*b_+4*c_+5*d_
                if y_>0:
                    y.append(1)
                else:
                    y.append(0)

index=list(range(len(x)))
import random
random.shuffle(index)


#train
ANN.ANN(x,y,w, train_iteration, learn_rate, 1)

#test
p=ANN.predictor([1,2,3,4], w)[-1] #should give a value close to 1, its label
print(p)

p=ANN.predictor([-1,-2,-3,-4], w)[-1] #should give a value close to 0, its label
print(p)

p=ANN.predictor([-1,-2,1,1], w)[-1] #should give a value close to 1, its label
print(p)
