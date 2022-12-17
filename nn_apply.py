import pickle
from nn import NN,load_data
import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':
    #load network
    with open("test.pkl","rb") as infile:
        nn = pickle.load(infile)

    print("loaded file")
    data_train,Y_train,data_test,Y_test = load_data()


    preds = nn.forward_pass(data_test)


    idx_rnd = np.random.choice(range(len(Y_test)),9)
    fig,axs = plt.subplots(3,3)
    ax_fl = np.flatten(axs)
    for i in range(9):
        ax_fl[i].imshow(data_test[idx_rnd])

    plt.show()




