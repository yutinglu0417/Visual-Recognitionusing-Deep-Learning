from dataloader import build_loader
from resnext import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import csv 

BATCH_SIZE = 128
LEARNING_RATE = 0.00005
NUM_EPOCH = 200
MODE = "train"
final_path = "./final_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCH)
best_path = "./best_" + str(BATCH_SIZE) + "_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCH)
load_path = "best_resnext+autoaug_128_5e-05_300(91.3333).pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

def plot_acc_loss(
        epochs,
        train_acc_list = None, 
        train_loss_list = None,
        valid_acc_list = None):

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(epochs, valid_acc_list, label='Valid Accuracy', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('accuracy_plot.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_list, label='Train Loss', marker='o', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()

def setup():
    # build up dataloader
    train_dl, val_dl, test_dl = build_loader(batch_size=BATCH_SIZE)

    # build up model
    model = resnext50(num_classes=100)
    model_state = model.state_dict()
    checkpoint = torch.load("./resnext50.pth")
    checkpoint_model = checkpoint['model_state_dict']
    pretrain_dict = {k:v for k, v in checkpoint_model.items() if k in model_state}
    model_state.update(pretrain_dict)
    model.load_state_dict(model_state)
    model = model.to(device)

    # print out number of model's parameter
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print("#Params:", num_params)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    return train_dl, val_dl, test_dl, model, criterion, optimizer, scheduler

def train_one_batch(
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        scheduler=None):    
       
    outputs, conloss_features = model(inputs)
    loss = criterion(outputs, targets)
    loss1 = con_loss(conloss_features, targets)
    total_loss = (loss+loss1)/2
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return total_loss.item(), outputs


def train(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer, 
        scheduler=None):
    
    model.train()
    total_loss = 0
    num_samples = 0
    num_correct = 0
    for images, labels in tqdm(dataloader, desc='train', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        loss, outputs = train_one_batch(model, criterion, optimizer, images, labels)
        total_loss += loss

        #accuracy        
        predicted = nn.Softmax(dim=1)(outputs)      
        _, predicted = torch.max(predicted, 1)
        num_correct += (predicted == labels).sum().item()
        num_samples += labels.size(0)

    acc = 100*num_correct/num_samples
    avgloss = total_loss/len(dataloader)
    return avgloss, acc 

def evaluate(
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader):
    
    model.eval()
    num_samples = 0
    num_correct = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="eval", leave=False):

            images, labels = images.to(device), labels.to(device)

            outputs, _ = model(images)

            #accuracy        
            predicted = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(predicted, 1)
            num_correct += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    acc = 100*num_correct/num_samples
    return acc

def test(model, dataloader):
    cvsname = "prediction.csv"
    csvfile = open(cvsname, 'w', newline='')
    w = csv.writer(csvfile)
    w.writerow(['image_name', 'pred_label'])
    model.eval()
    with torch.no_grad():
        for images, names in tqdm(dataloader, desc="test", leave=False):

            images = images.to(device)

            outputs, _ = model(images)

            #accuracy        
            predicted = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(predicted, 1)

            for i in range(images.size(0)):
                name = names[i].split('.')[0]
                w.writerow([name, predicted[i].item()])

def main():
    train_dl, valid_dl, test_dl, model, criterion, optimizer, scheduler = setup()
    if (MODE == "train"):
        best_acc = 0
        train_acc_list = []
        valid_acc_list = []
        train_loss_list = []
        for epoch_num in tqdm(range(1, NUM_EPOCH + 1)):
            loss, train_acc = train(model, train_dl, criterion, optimizer, scheduler)
            acc = evaluate(model, valid_dl)
            train_acc_list.append(train_acc)
            valid_acc_list.append(acc)
            train_loss_list.append(loss)
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    }, best_path + ".pth")
                
            print(f"epoch {epoch_num}: train loss: {loss:.4f}, train accuracy: {train_acc:.4f}, current accuracy: {acc:.4f}, best accuracy: {best_acc:.4f}")

        print(f"final accuracy: {acc:.4f}, best accuracy {best_acc:.4f}")
        torch.save({
                'model_state_dict': model.state_dict(),
                }, final_path + "_" + str(round(acc,4)) + ".pth")
        epochs = list(range(1, NUM_EPOCH+1)) 
        plot_acc_loss(epochs, train_acc_list, train_loss_list, valid_acc_list)

    elif (MODE == "test"):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test(model, test_dl)
    
    elif(MODE == "evaluate"):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, valid_dl)


if __name__ == "__main__":
    main()