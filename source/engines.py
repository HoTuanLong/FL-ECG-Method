import os, sys
from libs import *

def client_fit_fn(
    fit_loaders, client_num_epochs, 
    client_model, 
    optimizer, 
):
    print("\nStart Training ...\n" + " = "*16)
    client_model = client_model.cuda()
    client_metrics = {}

    for epoch in range(1, client_num_epochs + 1):
        print("epoch {}/{}".format(epoch, client_num_epochs) + "\n" + " - "*16)

        client_model.train()
        running_loss = 0.0
        running_tgts, running_prds = [], []
        for ecgs, tgts in tqdm.tqdm(fit_loaders["fit"]):
            ecgs, tgts = ecgs.cuda(), tgts.cuda()

            logits = client_model(ecgs)
            loss = F.binary_cross_entropy_with_logits(logits, tgts)

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, prds = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_prds.extend(prds)

        fit_loss, fit_f1 = running_loss/len(fit_loaders["fit"].dataset), metrics.f1_score(
            running_tgts, running_prds
            , average = "macro"
        )
        wandb.log(
            {
                "fit_loss":fit_loss, "fit_f1":fit_f1
            }, 
            step = epoch, 
        )
        client_metrics["fit_loss"], client_metrics["fit_f1"] = fit_loss, fit_f1

        with torch.no_grad():
            client_model.eval()
            running_loss = 0.0
            running_tgts, running_prds = [], []
            for ecgs, tgts in tqdm.tqdm(fit_loaders["evaluate"]):
                ecgs, tgts = ecgs.cuda(), tgts.cuda()

                logits = client_model(ecgs)
                loss = F.binary_cross_entropy_with_logits(logits, tgts)

                running_loss = running_loss + loss.item()*ecgs.size(0)
                tgts, prds = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() >= 0.5, 1.0, 0.0))
                running_tgts.extend(tgts), running_prds.extend(prds)

        evaluate_loss, evaluate_f1 = running_loss/len(fit_loaders["evaluate"].dataset), metrics.f1_score(
            running_tgts, running_prds
            , average = "macro"
        )
        wandb.log(
            {
                "evaluate_loss":evaluate_loss, "evaluate_f1":evaluate_f1
            }, 
            step = epoch, 
        )
        client_metrics["evaluate_loss"], client_metrics["evaluate_f1"] = evaluate_loss, evaluate_f1

    return client_metrics

def server_test_fn(
    test_loader, 
    server_model, 
):
    print("\nStart Testing ...\n" + " = "*16)
    server_model = server_model.cuda()

    with torch.no_grad():
        server_model.eval()
        running_loss = 0.0
        running_tgts, running_prds = [], []
        for ecgs, tgts in tqdm.tqdm(test_loader):
            ecgs, tgts = ecgs.cuda(), tgts.cuda()

            logits = server_model(ecgs)
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