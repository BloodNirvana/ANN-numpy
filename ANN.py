import numpy as np 

def initialize_weights(hidden_layer_structure, train_sample_dimensions):
    weights=[]
    for layer_index in range(len(hidden_layer_structure)):
        weights.append([])
        if layer_index==0:
            for unit in range(hidden_layer_structure[layer_index]):
                weights[layer_index].append(np.random.uniform(-0.1,0.1,size=train_sample_dimensions+1))
        else:
            for unit in range(hidden_layer_structure[layer_index]):
                previous_layer_unit_amount=hidden_layer_structure[layer_index-1]
                weights[layer_index].append(np.random.uniform(-0.1,0.1,size=previous_layer_unit_amount+1))
    return weights

def squared_error(target_value, predicted_value):
    return 0.5*sum(np.square(t - p) for t, p in zip(target_value, predicted_value))

def hidden_unit(input, weights):
    #summation=x1w1+x2w2+...+wn
    
    summation = np.dot(input, weights[:-1])+weights[-1]
    output=sigmoid_function(summation)
    return output

def sigmoid_function(net):
    return 1/(1+np.exp(-net))

def predictor(x, weights):
    output=[] #output value of each layer

    output_layer_weights=weights[-1]
    hidden_layer_weights=weights[:-1]

    #hidden layer start
    for layer_index in range(len(hidden_layer_weights)):
        output.append([])
               
        for unit in hidden_layer_weights[layer_index]:
            if layer_index==0:   
                output[layer_index].append(hidden_unit(x, unit))
            else:
                output[layer_index].append(hidden_unit(output[layer_index-1], unit))
    #hidden layer end

    #output layer start
    output.append([])
    for unit in output_layer_weights:
        #because defined output layer unit have same structure as hidden layer unit
        #so use hidden_unit as its unit
        output[-1].append(hidden_unit(output[-2], unit)) 
    '''
    for i in range(len(output[-1])):
        output[-1][i]=round(output[-1][i])'''
    #output layer end

    return output


def ANN(x, y, weights, iterations, learning_rate, momentum):
    # x mXn m:size of train sample, n:dimensions
    # y mX1 m:size of label, value=classification index
    alpha=momentum

    classification=[]
    for c in y:
        if c not in classification:
            classification.append(c) #classification= [0,1]
    
    target_value=[]
    
    for y_ in y:
        for c in classification:
            if y_==c:
                target_value_for_single_sample=np.zeros(len(classification))
                target_value_for_single_sample[classification.index(c)]=1
                target_value.append(target_value_for_single_sample) 
    #print(classification)
    #print(target_value)
    predicted_value=[predictor(x_, weights)[-1] for x_ in x] #predictor(x_, weights)[-1]: the last layer output value
    #reshape from [[a],[b]] to [a,b]
    predicted_value_reshape=np.reshape(predicted_value, len(predicted_value))      
    SE=squared_error(y, predicted_value_reshape)
    
    for i in range(iterations):
        print('iterations: ', i)
        #Backpropagation 
        layer_output=[predictor(x_, weights) for x_ in x]

        #predicted_value=[l[-1] for l in layer_output]
        #predicted_value_reshape=np.reshape(predicted_value, len(predicted_value))      
        #SE_new=squared_error(y, predicted_value_reshape)

        if True:#SE_new<=SE: #continue train  
            #SE=SE_new               
                 
            #SGD OR Adam depends on the batch size of x

            #output layer start  
            output_value=[l[-1] for l in layer_output]        
            for t, o in zip(target_value, output_value):
                for u in range(len(weights[-1])): #update each unit of output layer  
                    derivative = -(t[u]-o[u])*(1-o[u])*o[u]
                    sample_index = output_value.index(o) #the index of train data            
                    delta=[-learning_rate*derivative*xji for xji in layer_output[sample_index][-2]]
                    delta.append(-learning_rate*derivative*1) #append bias w*1
                    #print(delta)
                    weights[-1][u]=weights[-1][u]+delta           
            #output layer end
            
            #hidden layer start
            #update back forward h3->h2->h1
            layer_index=np.arange(len(weights[:-1])) #[0,1,2,3,4,5]->[0,1,2,3,4]
            sequence=np.flip(layer_index) #[4,3,2,1,0]
            
            #print('outputlayer done')


            hidden_layer_derivative=[]      
            for layer_index in sequence:
                #print('layer:', layer_index)                
                if layer_index==sequence[0]: #if index indicate to last hidden layer 
                    layer_output=[predictor(x_, weights) for x_ in x] #update whole nework after update any layer 
                    hidden_layer_derivative.append([])
                    output_value = [l[-2] for l in layer_output]
                    downstream_output_value=[l[-1] for l in layer_output]
                   
                    for t, o_j, o_k in zip(target_value, output_value, downstream_output_value):
                        hidden_layer_derivative[0].append([]) #append for each train sample
                        for u_j in range(len(weights[layer_index])): #u_j unit of this layer
                            downstream_sum = 0                       
                            for u_k in range(len(weights[-1])): ##u_k unit of output layer
                                derivative_downstream = -(t[u_k]-o_k[u_k])*(1-o_k[u_k])*o_k[u_k]
                                downstream_sum = downstream_sum + derivative_downstream*weights[-1][u_k][u_j]
                            
                            derivative = downstream_sum * o_j[u_j] * (1 - o_j[u_j])
                            sample_index = output_value.index(o_j) #the index of train data                                                      
                            hidden_layer_derivative[0][sample_index].append(derivative)
                            
                            input_value = layer_output[sample_index][-3]                   
                            delta=[-learning_rate*derivative*xji for xji in input_value]
                            delta.append(-learning_rate*derivative*1) #append bias w*1
                            weights[layer_index][u_j]=weights[layer_index][u_j]+delta         
                   
                elif layer_index!=sequence[-1] and layer_index!=sequence[0]:
                    superimposed_time=sequence[0]-layer_index
                    s=superimposed_time
                    layer_output=[predictor(x_, weights) for x_ in x] #update whole nework after update any layer 
                    hidden_layer_derivative.append([])
                    output_value = [l[-2-s] for l in layer_output]
                        
                    downstream_output_value=[l[-1-s] for l in layer_output]

                    for o_j, o_k in zip(output_value, downstream_output_value):
                        hidden_layer_derivative[s].append([]) #append for each train sample
                        for u_j in range(len(weights[layer_index])): #u_j unit of this layer
                            downstream_sum = 0                       
                            for u_k in range(len(weights[-1-s])): ##u_k unit of downstream layer
                                sample_index = output_value.index(o_j) #the index of train data
                                    
                                derivative_downstream = hidden_layer_derivative[s-1][sample_index][u_k]
                                downstream_sum = downstream_sum + derivative_downstream*weights[-1-s][u_k][u_j]
                            
                            derivative = downstream_sum * o_j[u_j] * (1 - o_j[u_j])
                            sample_index = output_value.index(o_j) #the index of train data                                                      
                            hidden_layer_derivative[s][sample_index].append(derivative)
                            
                            input_value = layer_output[sample_index][-3-s]                   
                            delta=[-learning_rate*derivative*xji for xji in input_value]
                            delta.append(-learning_rate*derivative*1) #append bias w*1
                            weights[layer_index][u_j]=weights[layer_index][u_j]+delta
                    
                else:
                    layer_output=[predictor(x_, weights) for x_ in x] #update whole nework after update any layer 
                    #hidden_layer_derivative.append([])
                    output_value = [l[0] for l in layer_output]
                        
                    downstream_output_value=[l[1] for l in layer_output]

                    for o_j, o_k in zip(output_value, downstream_output_value):
                        #hidden_layer_derivative[s].append([]) #append for each train sample
                        for u_j in range(len(weights[layer_index])): #u_j unit of this layer
                            downstream_sum = 0                       
                            for u_k in range(len(weights[1])): ##u_k unit of downstream layer
                                sample_index = output_value.index(o_j) #the index of train data
                                '''
                                print('1:',hidden_layer_derivative[-1])
                                print('sample index:',sample_index)
                                print('len hidden layer',len(hidden_layer_derivative))
                                print('x:',x)
                                print('2:',hidden_layer_derivative[-1][sample_index])
                                print('3:',hidden_layer_derivative[-1][sample_index][u_k])
                                '''
                                derivative_downstream = hidden_layer_derivative[-1][sample_index][u_k]
                                downstream_sum = downstream_sum + derivative_downstream*weights[1][u_k][u_j]
                            
                            derivative = downstream_sum * o_j[u_j] * (1 - o_j[u_j])
                            sample_index = output_value.index(o_j) #the index of train data                                                      
                            #hidden_layer_derivative[s][sample_index].append(derivative)
                            
                            input_value = x[sample_index]                   
                            delta=[-learning_rate*derivative*xji for xji in input_value]
                            delta.append(-learning_rate*derivative*1) #append bias w*1
                            weights[layer_index][u_j]=weights[layer_index][u_j]+delta
            #hidden layer end
            #print('Train: ',SE,'--->',SE_new)
            
        else:
            break
    return weights