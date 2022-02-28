import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    kfold = 5
    for k in range(kfold):
        with open("./model/MSD Cardiac/logs/fold%d_log.txt" % k, "rb") as file:
            history = pickle.load(file)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title("fold%d model loss pic" % k)
        plt.ylabel("avg dice loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"],loc="lower right")
        plt.savefig("./model/MSD Cardiac/logs/fold%d loss pic.jpg" % k)
        plt.show()