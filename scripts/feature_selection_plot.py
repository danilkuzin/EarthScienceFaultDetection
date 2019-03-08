import h5py
import matplotlib.pyplot as plt

with h5py.File('train_valid_hist.h5', 'r') as hf:
    val_loss = hf['history_val_loss'][:]
    val_acc = hf['history_val_acc'][:]
    loss = hf['loss'][:]
    acc = hf['acc'][:]

plt.plot(acc)
plt.plot(val_acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
