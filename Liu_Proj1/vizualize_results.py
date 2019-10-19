#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 02:50:13 2019

@author: osboxes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def draw_graph(filename, model_name):
    
    #This works when the whole pickle file was written in one go.
    #We wrote in a loop, so will need to read in a loop too.
    
    #results = pickle.load( open(filename, "rb") )
    #print(results)
    
    results_selected = []
    f = open(filename, "rb")
    count  = 0
    while True:
        try:
            data = pickle.load(f)
            #print("count:", count)
            #print(data)
            count += 1
            if data["name"] == model_name:
                results_selected.append(data)
                
        except EOFError:
            break
    f.close()
    
    #results_selected = [x for x in results if x["name"] == model_name]
    
    
    #plt.gca().set_color_cycle(['red', 'green', 'blue', 'magenta', 'olive', 'cyan'])
    
    
    plt.subplot(2,1,1)
    plt.title(model_name)
    
    legends = []
    for element in results_selected:
        param = element["params"]
        accuracy = element["accuracy"]
        epochs = [x for x in range(len(accuracy))]
        plt.plot(epochs, accuracy, marker = "o", markersize = 2)
        legends.append(param)
    
    plt.xlim(0, len(accuracy)+40)    
    plt.legend(legends, loc = "upper right")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    
    plt.subplot(2,1,2)
    legends = []
    for element in results_selected:
        param = element["params"]
        val_accuracy = element["val_accuracy"]
        epochs = [x for x in range(len(accuracy))]
        plt.plot(epochs, val_accuracy,  marker = "o", markersize = 2)
        legends.append(param)
    
    plt.xlim(0, len(accuracy)+40)
    plt.legend(legends, loc = "upper right")
    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    
    plt.show()
        
    
    pass



if __name__ == "__main__":
    #filename = "Results/" + "learning_rate_decay__2019_1_27_18_59_30__results.pk"
    filename = "Results/" + "momentum__2019_2_19_2_58_22_.pk"
    
#    model_name = "model1_deep"
#    draw_graph(filename, model_name)
    
#    model_name = "model2_local_no_weight_share"
#    draw_graph(filename, model_name)
    
    model_name = "model3_local_weight_share_basically_CNN"
    draw_graph(filename, model_name)