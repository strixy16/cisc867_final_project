import torch
import torchvision
from rfs_preprocessing import create_SMOTE, get_mean_and_std, get_imbalanced, removeSmallScans, separateClasses
from rfs_loaders import JoinDatasets, RFS_Loader, SMOTE_Loader, WaveletLoader
from rfs_models import InceptionV3, Net, ResNet, VGG, WaveNet
from rfs_learn import FocalLoss, get_accuracy, get_patient_acc, train, valid
from rfs_plots import aucPlot, imshow, lossPlot, waveshow
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import fnmatch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# save_path = '/lila/data/gonen/travis/projects/cholangio/Data/'
save_path = '/Users/katyscott/Documents/ICC/Data/'
# lib_path = save_path + 'chol_surv_files.csv'
lib_path = save_path + 'Labels/RFS_all_tumors_NaN.csv'
# data_path = '/lila/data/gonen/travis/projects/cholangio/Binary_files/NaN/'
data_path = '/Users/katyscott/Documents/ICC/Data/Images/Tumors/NaN/'
# looks like the path to wherever you want to save your output. Worry about this later
# rfs_path = '/lila/data/gonen/travis/projects/cholangio/recurrence/Results/'
rfs_path = '/Users/katyscott/Documents/ICC/Data/Results/'

nz_pix_limit = 300  # nz is number of zeros, setting limit for number of pixels in tumour in image
batch_size = 60
num_workers = 20  #
shuffle_dataset = True
data_split = 0.6
val = 1
random_seed = 30
epochs = 40
reg_type = 'L2'
lambda_reg = 0.001
lr = 1e-3
weight_decay = 1e-3
step_size = 10
gamma = 0.5
betas = [0.7, 0.85]
network = ['resnet18', 'inceptionv3', 'wavelet', 'net', 'vgg11_bn']
net_type = network[0]
im_dim = 224
data_aug = ['train', 'hflip', 'vflip', 'noise', 'crop', 'rotate', 'trans', 'bright',
            'train', 'hflip', 'vflip', 'noise', 'crop', 'rotate', 'trans', 'bright',
            'train', 'hflip', 'vflip', 'noise', 'crop', 'rotate', 'trans', 'bright']
load_type = ['regular','wave']

final_param = pd.DataFrame({'Pixel Limit': nz_pix_limit, 'Batch Size': batch_size, 'Split': data_split, 'Epochs': epochs,
                           'LR': lr, 'Weight Decay': weight_decay, 'Step Size': step_size, 'Gamma': gamma, 'betas': betas,
                            'Reg Type': reg_type, 'Lambda': lambda_reg, 'Network': net_type, 'Image Dim': im_dim})
final_param.T.to_csv(rfs_path + 'rfs_params.csv')

# Creating data indices for training & validation splits.
parse_indices = removeSmallScans(lib_path, data_path, nz_pix_limit)
sets = separateClasses(lib_path, parse_indices, data_split, val, random_seed)
if val == 1:
    train_indices, (val_indices, test_indices) = sets
else:
    train_indices, test_indices = sets
    val_indices = []

# Spamming the training with all one-labeled data to troubleshoot algorithm
imbalance_indices = get_imbalanced(lib_path, train_indices)

# SMOTE data
# imgs_smote, y_smote = create_SMOTE(lib_path, data_path, train_indices)
# print(imgs_smote.shape)
# print("SMOTE data created")

val_data, bal_data = [], []
if net_type == network[2]:
    channels = 4
    mean, std = list([0,0,0,0]), list([1,1,1,1])
    train_data = WaveletLoader(lib_path, data_path, train_indices, 'none', mean, std)
    mean, std = get_mean_and_std(train_data, channels)
    train_data = WaveletLoader(lib_path, data_path, train_indices, 'none', mean, std)
    test_data = WaveletLoader(lib_path, data_path, test_indices, 'none', mean, std)

    if val == 1:
        val_data = WaveletLoader(lib_path, data_path, val_indices, mean, 'none', std)

    synth_data = JoinDatasets(lib_path, data_path, train_indices,data_aug[:16], mean, im_dim, std,load_type[1])
else:
    channels = 3
    mean, std = list([0, 0, 0]), list([1, 1, 1])
    train_data = RFS_Loader(lib_path, data_path, train_indices, mean, std, im_dim)
    mean, std = get_mean_and_std(train_data, channels)

    train_data = RFS_Loader(lib_path, data_path, train_indices, mean, std, im_dim)
    test_data = RFS_Loader(lib_path, data_path, test_indices, mean, std, im_dim)
    imbal_data = RFS_Loader(lib_path, data_path, imbalance_indices, mean, std, im_dim)
    # smote_data = SMOTE_Loader(imgs_smote, y_smote, mean, std, im_dim)
    if val == 1:
        val_data = RFS_Loader(lib_path, data_path, val_indices, mean, std, im_dim)

    zero_data = JoinDatasets(lib_path, data_path, imbalance_indices, data_aug[:13], mean, std, im_dim, load_type[0])
    synth_data = JoinDatasets(lib_path, data_path, train_indices, data_aug[:3], mean, std, im_dim, load_type[0])
    bal_data = ConcatDataset((zero_data, train_data)) # balance out classes with augmentation
    # bal_smote = ConcatDataset((zero_data, synth_data, smote_data))
    # print(len(bal_smote))


synth_indices = list(range(len(synth_data)))
test_indices2 = list(range(len(test_data)))

if shuffle_dataset:
    np.random.seed(random_seed)
    # np.random.shuffle(train_indices)
    # np.random.shuffle(synth_indices)
    if val == 1: np.random.shuffle(val_indices)
    np.random.shuffle(test_indices2)

# Creating PT data samplers and loaders:
# synth_sampler = SubsetRandomSampler(synth_indices)
train_sampler = SubsetRandomSampler(train_indices)
if val == 1: val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices2)

train_loader = DataLoader(bal_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler)
if val == 1:
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    loader = val_loader
else:
    loader = test_loader

# get some random training images
dataiter = iter(train_loader)
images, labels, _, _ = dataiter.next()

# show images in a grid
if net_type == network[3]:
    images = images[:,1,:,:].unsqueeze(1)
    waveshow(torchvision.utils.make_grid(images), mean, std, rfs_path)
else:
    imshow(torchvision.utils.make_grid(images), mean, std, rfs_path)
print(labels)

# Select the proper network
if net_type == network[0]:
    net = ResNet(net_type).to(device)
elif net_type == network[1]:
    inceptionv3 = models.inception_v3(pretrained=True)
    inceptionv3.aux_logits = False
    num_ftrs = inceptionv3.fc.in_features
    # features = list(inceptionv3.classifier.children()[:-1])
    # features.extend([nn.Linear(num_ftrs, 500, bias=True)])
    # features.extend([nn.Linear(500, 20, bias=True)])
    # features.extend([nn.Linear(20, 2, bias=True)])
    # inceptionv3.classifier = nn.Sequential(*features)
    inceptionv3.fc = nn.Linear(num_ftrs, 2)
    net = inceptionv3.to(device)
    # net = InceptionV3(inceptionv3).to(device)
elif net_type == network[2]:
    net = WaveNet().to(device)
elif net_type == network[3]:
    net = Net().to(device)
elif net_type == network[4]:
    net = VGG(net_type).to(device)

print("Network set!")

# Define a loss function and optimizer
optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Train the network with validation set
train_losses, val_losses = [], []
f1score_train_tot, f1score_val_tot, loss_train_tot, loss_valid_tot, prob_tot = [], [], [], [], []
final_out = pd.DataFrame(columns=['Epoch', 'Train', 'Val', 'Pat Train', 'Pat Val'])
final_data = pd.DataFrame(columns=['ID', 'Label', 'Pred', 'Prob'])
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss()

for epoch in range(epochs):  # loop over the dataset multiple times
    loss_train, prob_train, labels_train, pred_train, id_train, slice_train = [], [], [], [], [], []
    correct = 0
    train_acc = 0
    scheduler.step()
    # training loop
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels, train_id, train_slice = data
        labels_train = np.hstack((labels_train, labels))
        id_train = np.hstack((id_train, train_id)) # id labels patients
        slice_train = np.hstack((slice_train, train_slice))
        inputs, labels, train_id, train_slice = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long), \
            train_id.to(device, dtype=torch.long), train_slice.to(device, dtype=torch.long)

        outputs, prob = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prob = prob.detach().cpu().numpy()
        prob_train = np.hstack((prob_train, prob[:,1]))
        pred_train = np.hstack((pred_train, np.array(predicted.cpu())))
        loss_train.append(train(net, inputs, labels, criterion, optimizer, reg_type, lambda_reg))
        train_acc += get_accuracy(outputs, labels)
    if epoch < 10:
        ep_str = '0' + str(epoch)
    else:
        ep_str = str(epoch)
    torch.save(net, rfs_path + 'model_' + ep_str + '.pth')
    loss_train_tot.append(sum(loss_train) / len(loss_train))
    pat_train_acc, pat_train_prob, pat_train_lb, pat_train_id = get_patient_acc(pred_train, labels_train, id_train, prob_train)
    f1score_train = f1_score(labels_train, pred_train, average='macro')
    f1score_train_tot.append(f1score_train)
    train_acc = 100 * train_acc / len(train_loader)

    loss_valid, prob_valid, labels_valid, pred_valid, id_valid, slice_valid = [], [], [], [], [], []
    val_correct = 0
    val_acc = 0
    # validation loop
    for i, data in enumerate(loader, 0):
        images, labels, val_id, val_slice = data
        labels_valid = np.hstack((labels_valid, labels))
        id_valid = np.hstack((id_valid, val_id))
        slice_valid = np.hstack((slice_valid, val_slice))
        images, labels, val_id, val_slice = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long), \
                                    val_id.to(device, dtype=torch.long), val_slice.to(device, dtype=torch.long)
        outputs, prob = net(images)
        _, predicted = torch.max(outputs.data, 1)
        prob = prob.detach().cpu().numpy()
        prob_valid = np.hstack((prob_valid, prob[:,1]))
        pred_valid = np.hstack((pred_valid, np.array(predicted.cpu())))
        loss_valid.append(valid(net, images, labels, criterion, reg_type, lambda_reg))
        val_acc += get_accuracy(outputs, labels)
    loss_valid_tot.append(sum(loss_valid)/len(loss_valid))
    pat_val_acc, pat_val_prob, pat_val_lb, pat_val_id = get_patient_acc(pred_valid, labels_valid, id_valid, prob_valid)
    f1score_val = f1_score(labels_valid, pred_valid, average='macro')
    f1score_val_tot.append(f1score_val)
    val_acc = 100 * val_acc / len(loader)
    if epoch % 10 == 0:
        item2 = pd.DataFrame({'ID': id_valid, 'Slice': slice_valid, 'Label': labels_valid, 'Pred': pred_valid, 'Prob': prob_valid})
        item2.to_csv(rfs_path + str(epoch) + '_rfs_val_data.csv')
        item2 = pd.DataFrame(columns=['ID', 'Slice', 'Label', 'Pred', 'Prob'])
        print("Total Training Patient Accuracy: %.3f" % pat_train_acc,
              "Total Val Patient Accuracy: %.3f" % pat_val_acc)
    print("Epoch: %d/%d" % (epoch+1, epochs),
          "Train Loss: %.3f" % (sum(loss_train)/len(loss_train)),
          "Train Accuracy: %.3f" % train_acc,
          "Val Loss: %.3f" % (sum(loss_valid)/len(loss_valid)),
          "Val Accuracy: %.3f" % val_acc
          )
    if epoch == 0:
        final_pat = pd.DataFrame({'Pat ID': pat_val_id, 'Pat Label': pat_val_lb})
    else:
        final_pat[str(epoch)] = pat_val_prob
    item = pd.DataFrame({'Epoch': [epoch], 'Train': [train_acc], 'Train F1': [f1score_train], 'Val': [val_acc],
                         'Val F1': [f1score_val], 'Pat Train': [pat_train_acc], 'Pat Val': [pat_val_acc]})
    final_out = pd.concat([final_out, item], sort='False')
final_pat.to_csv(rfs_path + 'rfs_pat.csv') # patient accuracy
final_out.to_csv(rfs_path + 'rfs_acc.csv') # slice accuracy
final_out = pd.DataFrame(columns=['Epoch', 'Train', 'Train F1', 'Val', 'Val F1', 'Pat Train', 'Pat Val'])

print('Finished Training')

lossPlot(loss_train_tot, loss_valid_tot, rfs_path)

# Keep only the best model according to f1 score
f1_max_idx = np.argmax(f1score_val_tot)
if f1_max_idx < 10:
    f1_str = '0' + str(f1_max_idx)
else:
    f1_str = str(f1_max_idx)

for file in os.listdir(rfs_path):
    if fnmatch.fnmatch(file, '*.pth'):
        if f1_str in file:
            net = torch.load(rfs_path + 'model_' + f1_str + '.pth')
        else:
            os.remove(rfs_path + file)

# Test the network on the test data
correct = 0
total = 0
net.eval()

# See accuracy for each class
classes = ('no recurrence', 'recurrence')
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
labels_tot, pred_tot, prob_tot, id_tot = [], [], [], []
roc_name, pat_name = 'roc.png', 'roc_pat.png'
with torch.no_grad():
    for data_test in test_loader:
        images, labels, id, slice = data_test
        labels_tot = np.hstack((labels_tot, labels))
        id_tot = np.hstack((id_tot, id))
        images, labels, id, slice = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long), \
                                    id.to(device, dtype=torch.long), slice.to(device, dtype=torch.long)

        outputs, prob = net(images)
        prob = prob.detach().cpu().numpy()
        _, predicted = torch.max(outputs.data, 1)
        prob_tot = np.hstack((prob_tot, prob[:,1]))
        pred_tot = np.hstack((pred_tot, np.array(predicted.cpu())))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels)
        for i in range(len(c)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

pat_test_acc, pat_test_prob, pat_test_lb, pat_test_id = get_patient_acc(pred_tot, labels_tot, id_tot, prob_tot)

precision, recall, fscore, support = precision_recall_fscore_support(labels_tot, pred_tot, average=None)
precision_pat, recall_pat, fscore_pat, support_pat = precision_recall_fscore_support(pat_test_lb, pat_test_prob.round(), average=None)

fpr, tpr, thresh = roc_curve(labels_tot, prob_tot)
fpr_pat, tpr_pat, thresh_pat = roc_curve(pat_test_lb, pat_test_prob)

tn, fp, fn, tp = confusion_matrix(labels_tot, pred_tot).ravel()
tn_pat, fp_pat, fn_pat, tp_pat = confusion_matrix(pat_test_lb, pat_test_prob.round()).ravel()

specificity0 = tp/(tp + fn)
specificity0_pat = tp_pat/(tp_pat + fn_pat)
specificity = tn/(tn + fp)
specificity_pat = tn_pat/(tn_pat + fp_pat)

npv0 = tp/(tp + fp)
npv0_pat = tp_pat/(tp_pat + fp_pat)
npv = tn/(tn + fn)
npv_pat = tn_pat/(tn_pat + fn_pat)

auc = roc_auc_score(labels_tot, prob_tot)
auc_pat = roc_auc_score(pat_test_lb, pat_test_prob)

print('Accuracy of the network on the test images: %.2f %%' % (
    100 * correct / total))
print("Total Test Patient Accuracy: %.2f %%" % pat_test_acc)
print('AUC: %.2f\n' % auc)
print('Patient AUC: %.2f\n' % auc_pat)
print('Sensitivity (no recurrence): %.2f %%\n' % (100 * recall[0]), 'Sensitivity (recurrence): %.3f %%' % (100 * recall[1]))
print('Patient Sensitivity (no recurrence): %.2f %%\n' % (100 * recall_pat[0]),
      'Patient Sensitivity (recurrence): %.2f %%' % (100 * recall_pat[1]))
print('Specificity (no recurrence): %.2f %%\n' % (100 * specificity0), 'Specificity (recurrence): %.3f %%' % (100 * specificity))
print('Patient Specificity (no recurrence): %.2f %%\n' % (100 * specificity0_pat),
      'Patient Specificity (recurrence): %.2f %%' % (100 * specificity_pat))
print('PPV (no recurrence): %.2f %%\n' % (100 * precision[0]), 'PPV (recurrence): %.3f %%' % (100 * precision[1]))
print('Patient PPV (no recurrence): %.2f %%\n' % (100 * precision_pat[0]),
      'Patient PPV (recurrence): %.2f %%' % (100 * precision_pat[1]))
print('NPV (no recurrence): %.2f %%\n' % (100 * npv0), 'NPV (recurrence): %.3f %%' % (100 * npv))
print('Patient NPV (no recurrence): %.2f %%\n' % (100 * npv0_pat),
      'Patient NPV (recurrence): %.2f %%' % (100 * npv_pat))
print('Confusion Matrix:\n', (confusion_matrix(labels_tot, pred_tot)))
print('Patient Confusion Matrix:\n', (confusion_matrix(pat_test_lb, pat_test_prob.round())))

aucPlot(auc, fpr, tpr, rfs_path, roc_name)
aucPlot(auc_pat, fpr_pat, tpr_pat, rfs_path, pat_name)

final_perf = pd.DataFrame({'Acc': correct / total, 'Patient Acc': pat_test_acc, 'AUC': auc, 'Patient AUC': auc_pat,
                           'Sn': recall, 'Patient Sn': recall_pat, 'Sp': [specificity0, specificity],
                           'Patient Sp': [specificity0_pat, specificity_pat], 'PPV': precision, 'Patient PPV': precision_pat,
                           'NPV': [npv0, npv], 'Patient NPV': [npv0_pat, npv_pat]})
final_perf.T.to_csv(rfs_path + 'rfs_perf.csv')
