import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

def client_fit_fn(
    fit_loaders, num_epochs, 
    client_model, 
    client_optim, 
    dataset,
    device = torch.device("cpu"), 
):
    print("\nStart Client Fitting ...\n" + " = "*16)
    client_model = client_model.to(device)

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)
        client_model.train()
        for ecgs, tgts in tqdm.tqdm(fit_loaders["fit"]):
            ecgs, tgts = ecgs.float().to(device), tgts.float().to(device)

            logits = client_model(ecgs)
            loss = sum([F.binary_cross_entropy_with_logits(logits[:, i], tgts[:, i]) for i in range(30)])

            loss.backward()
            client_optim.step(), client_optim.zero_grad()

    with torch.no_grad():
        client_model.eval()
        running_loss = 0.0
        running_tgts, running_predis,  = [], [], 
        for ecgs, tgts in tqdm.tqdm(fit_loaders["evaluate"]):
            ecgs, tgts = ecgs.float().to(device), tgts.float().to(device)

            logits = client_model(ecgs)
            loss = sum([F.binary_cross_entropy_with_logits(logits[:, i], tgts[:, i]) for i in range(30)])

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, predis = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() > 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_predis.extend(predis), 

    evaluate_loss, evaluate_f1 = running_loss/len(fit_loaders["evaluate"].dataset), metrics.f1_score(
        running_tgts, running_predis, 
        average = "macro", 
    )
    
    temp_models_dir = os.path.abspath(os.path.join(__dir__, "../temp_models"))
    os.makedirs(temp_models_dir, exist_ok=True)

    torch.save(client_model, os.path.join(temp_models_dir, "{}.ptl".format(dataset)))
    
    print("{:<8} - loss:{:.4f}, f1:{:.4f}".format("evaluate", 
        evaluate_loss, evaluate_f1
    ))

    print("\nFinish Client Fitting ...\n" + " = "*16)
    return {
        "evaluate_loss":evaluate_loss, "evaluate_f1":evaluate_f1
    }

def client_test_fn(
    test_loaders, 
    client_model, 
    device = torch.device("cpu"), 
):
    print("\nStart Client Testing ...\n" + " = "*16)
    client_model = client_model.to(device)

    with torch.no_grad():
        client_model.eval()
        running_loss = 0.0
        running_tgts, running_predis,  = [], [], 
        for ecgs, tgts in tqdm.tqdm(test_loaders["test"]):
            ecgs, tgts = ecgs.float().to(device), tgts.float().to(device)

            logits = client_model(ecgs)
            loss = sum([F.binary_cross_entropy_with_logits(logits[:, i], tgts[:, i]) for i in range(30)])

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, predis = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() > 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_predis.extend(predis), 

    test_loss, test_f1 = running_loss/len(test_loaders["test"].dataset), metrics.f1_score(
        running_tgts, running_predis, 
        average = "macro", 
    )
    print("{:<8} - loss:{:.4f}, f1:{:.4f}".format("test", 
        test_loss, test_f1
    ))

    print("\nFinish Client Testing ...\n" + " = "*16)
    return {
        "test_loss":test_loss, "test_f1":test_f1
    }
    
def server_val_fn(
    fit_loaders, 
    server_model, 
    device = torch.device("cpu"), 
):
    server_model = server_model.to(device)
    with torch.no_grad():
        server_model.eval()
        running_loss = 0.0
        running_tgts, running_predis,  = [], [], 
        for ecgs, tgts in tqdm.tqdm(fit_loaders["evaluate"]):
            ecgs, tgts = ecgs.float().to(device), tgts.float().to(device)

            logits = server_model(ecgs)
            loss = sum([F.binary_cross_entropy_with_logits(logits[:, i], tgts[:, i]) for i in range(30)])

            running_loss = running_loss + loss.item()*ecgs.size(0)
            tgts, predis = list(tgts.data.cpu().numpy()), list(np.where(torch.sigmoid(logits).detach().cpu().numpy() > 0.5, 1.0, 0.0))
            running_tgts.extend(tgts), running_predis.extend(predis), 

    evaluate_loss, evaluate_f1 = running_loss/len(fit_loaders["evaluate"].dataset), metrics.f1_score(
        running_tgts, running_predis, 
        average = "macro", 
    )
    return {
        "evaluate_loss":evaluate_loss, "evaluate_f1":evaluate_f1
    }