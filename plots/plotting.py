
def plot_loss():
    plt.plot(history.history['loss'],'r',label='training loss')
    plt.plot(history.history['val_loss'],label='validation loss')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_loss()
    
def plot_acc():
    plt.plot(history.history['accuracy'],'r',label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='validation accuracy')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
plot_acc()


def evaluate_model():
    score = model.evaluate(X_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
evaluate_model()

