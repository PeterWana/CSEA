import CSEA
'''
Set the CSEA parameters and run the algorithm here.

The first parameter is the number of clusters, for example there are 2 clusters in the karate network.
The second parameter is the neural network layers and neurons, for example, "24, 18" means the neural network has two layers,
the number of neurons in the first layer is 24, and the number of neurons in the second layer is 18.
The third parameter is the loop count, for example, "2" means 2 runs on this dataset.
The fourth, fifth and sixth parameters are the Learning rate, Epochs and Batch size of the neural network, respectively.
In addition, please note that this experiment is conducted under Windows, so please try to use Windows system to run this experiment.
'''
filename = 'karate'
clusters = 2
D = [24, 18]
loop_count = 2
lr = 0.014
epochs = 100
batch_size = 256
res = CSEA.CSEA(filename, clusters, D, loop_count, lr, epochs, batch_size)
print(res)