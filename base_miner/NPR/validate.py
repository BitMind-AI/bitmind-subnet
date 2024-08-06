import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions


def validate(model, dataloader, device='cuda'):

    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in dataloader:
            keep_idx = [i for i, b in enumerate(batch) if b[0].shape[0] == batch[0][0].shape[0]]
            # batch = np.array(batch)
            inputs = torch.stack([b[0] for i, b in enumerate(batch) if i in keep_idx])
            labels = torch.stack([torch.tensor(b[1]) for i, b in enumerate(batch) if i in keep_idx])
            img, label = inputs.to(device).float(), labels.to(device).float()

            #in_tens = img.cuda()
            out = model(img).sigmoid().flatten().tolist()
            y_pred.extend(out)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
