import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


# Creates function to show an image
def imshow(img, mean, std, path):
    inv_normalize = transforms.Normalize(
        mean=-mean/std,
        std=1/std
    )
    img = inv_normalize(img)     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.savefig(path + 'sample_imgs.png')
    plt.clf()


def waveshow(img, mean, std, path):
    inv_normalize = transforms.Normalize(
        mean=-mean/std,
        std=1/std
    )
    img = inv_normalize(img)     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    # npimg = np.stack((npimg, npimg, npimg), axis=2)
    plt.imshow(npimg, interpolation='nearest')
    plt.savefig(path + 'wave_imgs.png')
    plt.clf()


def lossPlot(train, valid, path):
    plt.plot(train, label='Training loss')
    plt.plot(valid, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(path + 'training.png')
    plt.clf()


def aucPlot(auc, fpr, tpr, path, fname):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path + fname)
    plt.clf()