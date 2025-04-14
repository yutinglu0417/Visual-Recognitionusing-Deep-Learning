from dataloader import *
import torch
import torchvision.models as models
from tqdm.auto import tqdm
import csv
import torchmetrics

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCH = 15
MODE = "train"
THRESHOLD = 0.8
save_path = "epoch_"
final_path = "final"
best_path = "best"
load_path = "best_v2_aug_smallratio_sharp_0.4734_0.8440.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


def setup():
    # build up dataloader
    train_dl, val_dl, test_dl = build_loader(batch_size=BATCH_SIZE)

    # build up model
    anchor_sizes = ((32,), (64,), (96,), (128,), (160,))
    aspect_ratios = ((0.33, 0.5, 0.67),) * len(anchor_sizes)
    anchor_generator = models.detection.faster_rcnn.AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )
    model.rpn.anchor_generator = anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes=11
        )
    )
    model = model.to(device)

    # print out number of model's parameter
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print("#Params:", num_params)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    map_metric = torchmetrics.detection.MeanAveragePrecision().to(device)
    return train_dl, val_dl, test_dl, model, optimizer, scheduler, map_metric


def train_one_batch(model, optimizer, inputs, targets, scheduler=None):
    loss_dict = model(inputs, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return losses


def train(model, dataloader, optimizer, scheduler=None):

    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader, desc='train', leave=False):
        images_list = list(img.to(device) for img in images)
        targets_list = [{k: v.to(device) for k, v in t.items()}
                        for t in targets]
        loss = train_one_batch(model, optimizer, images_list, targets_list)

        total_loss += loss

    avgloss = total_loss/len(dataloader)
    return avgloss


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
        for images, targets in tqdm(
            data_loader, desc="Evaluating", leave=False
        ):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            output = model(images)

            boxes = [t['boxes'].to(device) for t in targets]
            labels = [t['labels'].to(device) for t in targets]

            map_metric.update(output, targets)

            for i, out in enumerate(output):
                true_label = targets[i]["labels"].tolist()
                if (true_label == []):
                    true_label = -1
                else:
                    true_label = [label-1 for label in true_label]
                    true_label = int("".join(map(str, true_label)))

                boxes = out["boxes"].tolist()
                scores = out["scores"].tolist()
                predict_labels = out["labels"].tolist()

                # 篩選符合條件的預測
                filtered_results = [
                    (box, score, label)
                    for box, score, label in zip(boxes, scores, predict_labels)
                    if score > THRESHOLD
                ]

                # 按照 x_min 進行排序（確保左至右）
                filtered_results.sort(key=lambda x: x[0][0])

                # 輸出數字結果
                sorted_labels = [label for _, _, label in filtered_results]
                sorted_labels = [label-1 for label in sorted_labels]
                if sorted_labels == []:
                    sorted_labels = -1
                else:
                    sorted_labels = int("".join(map(str, sorted_labels)))

                if (true_label == sorted_labels):
                    num_correct += 1

                index += 1

            num_total += len(output)
            # 遍歷所有圖片的推理結果
            for i, out in enumerate(output):
                image_name = str(targets[i]["image_id"].item()) + '.png'
                image_path = os.path.join(input_folder, image_name)
                output_path = os.path.join(output_folder, image_name)

                # 讀取原圖
                image = cv2.imread(image_path)
                if image is None:
                    print(f"無法讀取圖片: {image_path}")
                    continue

                # 轉換 Tensor 為 Python 列表
                boxes = out["boxes"].tolist()
                scores = out["scores"].tolist()
                labels = out["labels"].tolist()

                # 過濾符合閾值的框
                for box, score, label in zip(boxes, scores, labels):
                    if score > THRESHOLD:
                        x_min, y_min, x_max, y_max = map(int, box)  # 轉為整數座標
                        # 繪製紅框
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                                      (0, 0, 255), 2)
                        # 標註數字
                        text = f"{label-1} ({score:.2f})"
                        cv2.putText(image, text, (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3, (0, 0, 255), 1)

                # 儲存標註後的圖片
                cv2.imwrite(output_path, image)
                print(f"已儲存標註圖片: {output_path}")

    _mAP = map_metric.compute()
    print(f"Validation mAP: {_mAP['map']:.4f}")
    acc = num_correct/num_total
    print(f"Validation acc: {acc:.4f}")
    return _mAP['map'], acc


def test(model, dataloader):
    cvsname = "prediction.csv"
    csvfile = open(cvsname, 'w', newline='')
    w = csv.writer(csvfile)
    w.writerow(['image_id', 'pred_label'])
    image_results = []
    index = 1
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="testing", leave=False):
            images = [img.to(device) for img in images]
            output = model(images)

            for i, out in enumerate(output):
                boxes = out["boxes"].tolist()
                scores = out["scores"].tolist()
                labels = out["labels"].tolist()

                for box, score, label in zip(boxes, scores, labels):
                    if score > THRESHOLD:
                        x_min, y_min, x_max, y_max = box
                        image_results.append({
                            "image_id": targets[i],
                            "bbox": [
                                x_min,
                                y_min,
                                x_max - x_min,
                                y_max - y_min
                            ],
                            "score": float(score),
                            "category_id": int(label)
                        })

                # 篩選符合條件的預測
                filtered_results = [
                    (box, score, label)
                    for box, score, label in zip(boxes, scores, labels)
                    if score > THRESHOLD
                ]

                # 按照 x_min 進行排序（確保左至右）
                if (filtered_results != []):
                    filtered_results.sort(key=lambda x: x[0][0])

                    # 輸出數字結果
                    sorted_labels = [
                        label - 1
                        for _, _, label in filtered_results
                    ]
                    sorted_labels = int("".join(map(str, sorted_labels)))
                else:
                    sorted_labels = -1
                index += 1
                w.writerow([targets[i], sorted_labels])

    # 轉換為 JSON 並儲存
    output_json = json.dumps(image_results, indent=4)

    with open("output.json", "w") as f:
        f.write(output_json)


def main():
    train_dl, valid_dl, test_dl, model, optimizer, scheduler, map_metric = \
        setup()

    if (MODE == "train"):
        best_map = 0
        best_acc = 0
        for epoch_num in tqdm(range(1, NUM_EPOCH + 1)):
            loss = train(model, train_dl, optimizer, scheduler)
            val_map, val_acc = evaluate(model, valid_dl, map_metric)
            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with mAP: {best_map:.4f}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    }, best_path + str(epoch_num+1) + "mAP.pth")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with acc: {best_acc:.4f}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    }, best_path + str(epoch_num+1) + "acc.pth")

            print(f"epoch {epoch_num}: train loss: {loss:.4f}, "
                  f"current mAP: {val_map:.4f}, best mAP: {best_map:.4f}")
            print(f"current acc: {val_acc:.4f}, best acc: {best_acc:.4f}")

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
