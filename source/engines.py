import os, sys
from libs import *

def train_fn(
    train_loaders, num_epochs, 
    model, 
    optimizer, 
    scheduler, 
    save_ckp_dir = "./", 
):
    print("\nStart Training ...\n" + " = "*16)
    model = model.cuda()

    best_f1 = 0.0
    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        model.train()
        running_loss = 0.0
        running_tgts, running_prds = [], []
        for ecgs, tgts in tqdm.tqdm(train_loaders["train"]):
            ecgs, tgts = ecgs.cuda(), tgts.cuda()

            logits = model(ecgs)
            loss = F.binary_cross_entropy_with_logits(logits, tgts)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, prds = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_prds.extend(prds)

        train_loss, train_f1 = running_loss/len(train_loaders["train"].dataset), metrics.f1_score(
            running_tgts, running_prds
            , average = "macro"
        )
        wandb.log(
            {
                "train_loss":train_loss, "train_f1":train_f1
            }, 
            step = epoch, 
        )

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_tgts, running_prds = [], []
            for ecgs, tgts in tqdm.tqdm(train_loaders["val"]):
                ecgs, tgts = ecgs.cuda(), tgts.cuda()

                logits = model(ecgs)
                loss = F.binary_cross_entropy_with_logits(logits, tgts)

                running_loss = running_loss + loss.item()*ecgs.size(0)
                tgts, prds = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1.0, 0.0))
                running_tgts.extend(tgts), running_prds.extend(prds)

        val_loss, val_f1 = running_loss/len(train_loaders["val"].dataset), metrics.f1_score(
            running_tgts, running_prds
            , average = "macro"
        )
        wandb.log(
            {
                "val_loss":val_loss, "val_f1":val_f1
            }, 
            step = epoch, 
        )
        if val_f1 > best_f1:
            best_f1 = val_f1; torch.save(
                model, 
                "{}/best.ptl".format(save_ckp_dir), 
            )

        if (scheduler is not None) and (not epoch > scheduler.T_max):
            scheduler.step()

def test_fn(
    test_loader, 
    model, 
):
    print("\nStart Testing ...\n" + " = "*16)
    model = model.cuda()

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_tgts, running_prds = [], []
        for ecgs, tgts in tqdm.tqdm(test_loader):
            ecgs, tgts = ecgs.cuda(), tgts.cuda()

            logits = model(ecgs)
            loss = F.binary_cross_entropy_with_logits(logits, tgts)

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, prds = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_prds.extend(prds)

    test_loss, test_f1 = running_loss/len(test_loader.dataset), metrics.f1_score(
        running_tgts, running_prds
        , average = "macro"
    )
    print(
        "test_loss:{:.4f}".format(test_loss), "test_f1:{:.4f}".format(test_f1)
    )