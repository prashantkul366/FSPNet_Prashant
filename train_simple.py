import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import dataset
import FSPNet_model
import loss
import os


# ---------- Dice ----------
def dice_score(pred, mask):

    # pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum()

    return (2*inter + 1e-6)/(union + 1e-6)


# # ---------- Validation ----------
# def validate(model, loader):

#     model.eval()
#     dices = []

#     with torch.no_grad():
#         for b in loader:

#             img = b["img"].cuda(non_blocking=True)
#             mask = b["label"].cuda(non_blocking=True)

#             out = model(img)[-1]

#             d = dice_score(out, mask)
#             dices.append(d.item())

#     model.train()
#     return sum(dices)/len(dices)
def validate(model, loader):

    model.eval()

    dices = []

    print("\n------ VALIDATION START ------")

    with torch.no_grad():

        for i, b in enumerate(loader):

            img = b["img"].cuda(non_blocking=True)
            mask = b["label"].cuda(non_blocking=True)

            out = model(img)[-1]

            d = dice_score(out, mask)

            dices.append(d.item())

            if i % 50 == 0:
                print(
                    f"Val iter {i}/{len(loader)}  "
                    f"dice {d.item():.4f}  "
                    f"pred_min {out.min().item():.3f}  "
                    f"pred_max {out.max().item():.3f}"
                )

    mean_dice = sum(dices)/len(dices)

    print("------ VALIDATION END ------")
    print(f"Mean Dice : {mean_dice:.4f}\n")

    model.train()

    return mean_dice

def main():

    print("\n================ TRAIN START ================\n")

    torch.backends.cudnn.benchmark = True

    train_root = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_npy/train"
    val_root   = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_npy/val"

    print("Train path :", train_root)
    print("Val path   :", val_root)

    train_dataset = dataset.TrainDataset(train_root)
    val_dataset   = dataset.TrainDataset(val_root)

    print("\nTrain size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # ---------- sanity ----------
    b = next(iter(train_loader))
    print("\n========= SANITY =========")
    print("Input shape :", b["img"].shape)
    print("Mask shape  :", b["label"].shape)
    print("Img min/max :", b["img"].min().item(), b["img"].max().item())
    print("Mask unique :", torch.unique(b["label"]))
    print("==========================\n")

    # ---------- model ----------
    print("Loading model...")

    model = FSPNet_model.Model(
        "/content/drive/MyDrive/Prashant/Pretrain/deit_base_distilled_patch16_384.pth",
        img_size=384
    )

    total_params = sum(p.numel() for p in model.parameters())/1e6
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

    print(f"Total params : {total_params:.2f} M")
    print(f"Train params : {train_params:.2f} M")

    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    print("Optimizer : AdamW")
    print("LR        :", 3e-5)

    best_dice = 0
    patience = 50
    no_improve = 0

    print("\n=========== TRAIN LOOP ===========\n")

    for epoch in range(200):

        t0 = time.time()
        running = 0

        for i, b in enumerate(train_loader):

            img = b["img"].cuda(non_blocking=True)
            mask = b["label"].cuda(non_blocking=True)

            out = model(img)

            all_loss, main_loss = loss.multi_bce(out, mask)

            optimizer.zero_grad()
            all_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()

            running += main_loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch}  Iter {i}/{len(train_loader)}  loss {main_loss.item():.4f}")

        val_dice = validate(model, val_loader)

        epoch_time = time.time() - t0

        print(f"\n✅ Epoch {epoch}")
        print(f"Train loss : {running/len(train_loader):.4f}")
        print(f"Val Dice   : {val_dice:.4f}")
        print(f"Epoch time : {epoch_time:.1f} sec")

        if val_dice > best_dice:

            best_dice = val_dice
            no_improve = 0

            torch.save(model.state_dict(),"best_model.pth")

            print("⭐ BEST MODEL SAVED")
            print("⭐ Best Dice :", best_dice)

        else:
            no_improve += 1
            print("No improve count :", no_improve)

        print("----------------------------------")

        if no_improve >= patience:
            print("\n🛑 EARLY STOPPING TRIGGERED")
            break

    print("\n============= TRAIN END =============\n")


if __name__ == "__main__":
    main()