import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import dataset
import FSPNet_model
import loss


# ---------- Dice ----------
def dice_score(pred, mask):

    pred = (pred > 0.5).float()

    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()

    return (2*inter + 1e-6)/(union + 1e-6)


# ---------- Validation ----------
def validate(model, loader):

    model.eval()
    dices = []

    with torch.no_grad():
        for b in loader:

            img = b["img"].cuda()
            mask = b["label"].cuda()

            out = model(img)[-1]

            d = dice_score(out, mask)
            dices.append(d.item())

    model.train()
    return sum(dices)/len(dices)


def main():

    torch.backends.cudnn.benchmark = True

    train_root = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_npy/train"
    val_root   = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_npy/val"

    train_dataset = dataset.TrainDataset(train_root)
    val_dataset   = dataset.TrainDataset(val_root)

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ---------- sanity ----------
    b = next(iter(train_loader))
    print("\nSANITY")
    print("img:", b["img"].shape)
    print("mask:", b["label"].shape)
    print("mask unique:", torch.unique(b["label"]))
    print("------------\n")

    # ---------- model ----------
    model = FSPNet_model.Model(
        "/content/drive/MyDrive/Prashant/Pretrain/deit_base_distilled_patch16_384.pth",
        img_size=384
    )

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    best_dice = 0
    patience = 50
    no_improve = 0

    for epoch in range(200):

        t0 = time.time()

        running = 0

        for b in train_loader:

            img = b["img"].cuda()
            mask = b["label"].cuda()

            out = model(img)

            all_loss, main_loss = loss.multi_bce(out, mask)

            optimizer.zero_grad()
            all_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()

            running += main_loss.item()

        val_dice = validate(model, val_loader)

        print(f"Epoch {epoch}  loss {running/len(train_loader):.4f}  dice {val_dice:.4f}  time {(time.time()-t0):.1f}s")

        if val_dice > best_dice:

            best_dice = val_dice
            no_improve = 0

            torch.save(model.state_dict(),"best_model.pth")
            print("⭐ best saved")

        else:
            no_improve += 1

        if no_improve >= patience:
            print("EARLY STOP")
            break


if __name__ == "__main__":
    main()