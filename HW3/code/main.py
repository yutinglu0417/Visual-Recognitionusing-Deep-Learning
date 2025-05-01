from dataloader import *
import torch
import torchvision
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
from utils import encode_mask, decode_maskobj
from torchvision.models.detection.rpn import AnchorGenerator

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCH = 50
MODE = "train"
THRESHOLD = 0.8
NUM_CLASSES = 5
save_path = "epoch_"
final_path = "final"
best_path = "best"
load_path = "best_ver3_0.4539.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


class AdvancedMaskPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//4, num_classes, 1)
        )

    def forward(self, x):
        return self.layers(x)


def plot_and_save_curve(values, ylabel, title, save_path, color='blue'):
    epochs = list(range(1, len(values) + 1))

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif isinstance(values[0], torch.Tensor):
        values = [v.detach().cpu().item() for v in values]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values, marker='o', label=ylabel, color=color)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def setup():
    # build up dataloader
    train_dl, val_dl, test_dl = get_loader(batch_size=BATCH_SIZE)

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # build up model
    model = models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            NUM_CLASSES)

    model.rpn.anchor_generator = anchor_generator

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = \
        AdvancedMaskPredictor(in_features_mask,
                              hidden_dim=256,
                              num_classes=NUM_CLASSES)

    model.rpn.box_similarity = torchvision.ops.box_iou  # 使用IoU
    model.rpn.box_fg_iou_thresh = 0.4
    model.rpn.box_bg_iou_thresh = 0.1
    model.rpn.box_batch_size_per_image = 512  # 提高sample數量
    model.rpn.box_positive_fraction = 0.7
    # model.rpn.anchor_generator = anchor_generator
    model = model.to(device)
    # print(model.rpn.pre_nms_top_n_train)
    # print out number of model's parameter
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print("#Params:", num_params)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    map_metric = torchmetrics.detection.MeanAveragePrecision().to(device)
    return train_dl, val_dl, test_dl, model, optimizer, scheduler, map_metric


def train_one_batch(model, optimizer, inputs, targets, scheduler=None):
    loss_dict = model(inputs, targets)
    losses = sum(loss for loss in loss_dict.values())
    # print("Loss breakdown:", loss_dict)
    # print("Total loss:", losses.item())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return losses


def train(model, dataloader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    for images, targets, name in tqdm(dataloader, desc='train', leave=False):
        images_list = list(img.to(device) for img in images)
        targets_list = \
            [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = train_one_batch(model, optimizer, images_list, targets_list)

        total_loss += loss

    avgloss = total_loss/len(dataloader)
    return avgloss


def show_masks_on_image(image, masks, boxes):
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = img_np - img_np.min()
    img_np = img_np / img_np.max()

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)

    for i in range(len(masks)):
        mask = masks[i, 0].cpu().numpy()
        box = boxes[i].cpu().numpy()

        # 把 mask 畫成輪廓
        plt.contour(mask, levels=[0.5], colors='r', linewidths=1)

        # 畫框
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor='blue', facecolor='none', linewidth=2
        ))

    plt.axis("off")
    plt.show()


def evaluate(model, data_loader, map_metric):
    model.eval()

    index = 1
    input_folder = "data/valid/"  # 原圖目錄
    output_folder = "result/"
    os.makedirs(output_folder, exist_ok=True)
    num_correct = 0
    num_total = 0
    map_metric.reset()
    with torch.no_grad():
        for images, targets, names in tqdm(data_loader,
                                           desc="Evaluating",
                                           leave=False):
            images = [img.to(device) for img in images]
            targets = \
                [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(images)
            map_metric.update(output, targets)

    _mAP = map_metric.compute()
    print(f"Validation mAP: {_mAP['map_50']:.4f}")
    return _mAP['map_50']


def test(model, dataloader):
    results = []
    model.eval()
    with torch.no_grad():
        for images, ids, names in tqdm(dataloader,
                                       desc="testing",
                                       leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                image_id = ids[i]
                output = outputs[i]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for j in range(len(boxes)):
                    score = scores[j]
                    if score < 0.0:
                        continue

                    x1, y1, x2, y2 = boxes[j]
                    bbox = \
                        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                    binary_mask = masks[j, 0] > 0.5
                    rle = encode_mask(binary_mask)

                    result = {
                        "image_id": int(image_id),
                        "bbox": bbox,
                        "score": float(score),
                        "category_id": int(labels[j]),
                        "segmentation": rle
                    }
                    results.append(result)

    # 轉換為 JSON 並儲存
    output_json = json.dumps(results, indent=4)

    with open("output.json", "w") as f:
        f.write(output_json)


def main():
    train_dl, valid_dl, test_dl, model, optimizer, scheduler, map_metric = \
        setup()
    losses = []
    ap50 = []
    if (MODE == "train"):
        best_ap50 = 0
        for epoch_num in tqdm(range(1, NUM_EPOCH + 1)):
            loss = train(model, train_dl, optimizer, scheduler)
            scheduler.step()
            val_ap50 = evaluate(model, valid_dl, map_metric)
            losses.append(loss)
            ap50.append(val_ap50)
            if val_ap50 > best_ap50:
                best_ap50 = val_ap50
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with AP50: {best_ap50:.4f}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    }, best_path + str(epoch_num+1) + "AP50.pth")

            print(f"epoch {epoch_num}: train loss: {loss:.4f}")
            print(f"current AP50: {val_ap50:.4f}, best AP50: {best_ap50:.4f}")
            # print(f"current acc: {val_acc:.4f}, best acc: {best_acc:.4f}")
        plot_and_save_curve(losses, "Loss", title="Loss",
                            save_path="./loss.png", color="red")
        plot_and_save_curve(ap50, "AP50", title="AP50",
                            save_path="./ap.png", color="red")
        torch.save({
                'model_state_dict': model.state_dict(),
                }, final_path + ".pth")

    elif (MODE == "test"):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test(model, test_dl)

    elif (MODE == "evaluate"):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, valid_dl, map_metric)


if __name__ == "__main__":
    main()
