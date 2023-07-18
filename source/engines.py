import os, sys
from libs import *

def client_fit_fn(
    fit_loaders, client_num_epochs, 
    client_model, 
    optimizer, 
):
    print("\nStart Training ...\n" + " = "*16)
    client_model = client_model.cuda()

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
        print(
            "fit_loss:{:.4f}".format(fit_loss), "fit_f1:{:.4f}".format(fit_f1)
        )

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
        print(
            "evaluate_loss:{:.4f}".format(evaluate_loss), "evaluate_f1:{:.4f}".format(evaluate_f1)
        )

        client_results = {}
        client_results["fit_loss"], client_results["fit_f1"] = fit_loss, fit_f1
        client_results["evaluate_loss"], client_results["evaluate_f1"] = evaluate_loss, evaluate_f1

        return client_results