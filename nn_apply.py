import pickle
from nn import NN,load_data,DenseLayer
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")



class_names = ["T-shirt/top" , "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


if __name__ == '__main__':
    #load network
    with open("test.pkl","rb") as infile:
        nn = pickle.load(infile)

    print("loaded file")
    data_train,Y_train,data_test,Y_test,enc = load_data()

    preds = nn.predict(data_test.T).T

    acc = accuracy_score(np.argmax(preds, axis=1), np.argmax(Y_test.T, 1))

    conf_mat = confusion_matrix(np.argmax(Y_test, axis=0), np.argmax(preds.T, axis=0))

    fig,ax = plt.subplots(1)
    sns.heatmap(conf_mat)

    idx_rnd = np.random.choice(range(Y_test.shape[1]), 9)
    fig, axs = plt.subplots(3, 3)
    ax_fl = axs.flatten()
    for i in range(9):
        ax_fl[i].imshow(data_test[idx_rnd[i]].reshape(28, 28),
                        cmap = "gray")
        true_label = class_names[int(enc.inverse_transform(Y_test[:,idx_rnd[i]].reshape(1,-1))[0][0])]
        pred_label = class_names[np.argmax(preds[idx_rnd[i],:])]

        ax_fl[i].set_title("True Label: {}, predicted: {}".format(true_label,pred_label))
        ax_fl[i].axis("off")
    plt.tight_layout()

    plt.show()




