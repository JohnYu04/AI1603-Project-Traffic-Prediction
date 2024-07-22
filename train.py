import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from data_loader import *
from util import *
from model import *

if __name__ == '__main__':
    def vali(model, vali_loader, criterion, device):
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()

                outputs = model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()
        return total_loss

    seed = 21
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    train_loader = get_dataloader(data_path='train.csv')
    val_loader = get_dataloader(data_path='val.csv')
    test_loader = get_dataloader(data_path='test_history.csv')
    train_steps = len(train_loader)

    seq_len = 24
    n_hidden = 64
    n_layers = 2
    n_heads = 4
    pred_len = 1
    lr = 1e-3
    epochs = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = '.'

    adj_matrix = extract_208x208_matrix('paris_adj.csv')
    model = TransformerWithGAT(seq_len=seq_len, pred_len=pred_len, n_hidden=n_hidden, n_layers=n_layers,
                           n_heads=n_heads, dropout=0.1, adj=adj_matrix, n_heads_gat=8, alpha=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-15)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        model.train()
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            n_batch, n_vars = batch_y.shape[0], batch_y.shape[2]
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                iter_count = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        train_loss = np.average(train_loss)
        vali_loss = vali(model, val_loader, criterion, device)
        #test_loss = vali(model, test_loader, criterion, device)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Generate Test
    preds = None
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
    model.eval()
    train_loss = vali(model, train_loader, criterion, device)
    vali_loss = vali(model, val_loader, criterion, device)
    print("Final: Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(
        train_loss, vali_loss))
    for i, (batch_x) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)

        outputs = model(batch_x)

        pred = outputs.detach().cpu()
        if preds is None:
            preds = pred.squeeze(-1)
        else:
            preds = torch.cat((preds, pred.squeeze(-1)), dim=0)

    preds = preds.numpy()
    test_true = pd.read_csv(os.path.join('format.csv'))
    test_preds = pd.DataFrame(preds, columns=[test_true.columns[1:]])

    assert len(test_true) == len(test_preds), "Row counts of test_true and test_preds must match"
    test_preds.insert(0, 'date', test_true['date'])
    true_columns = test_true.columns
    test_preds.columns = true_columns

    output_file = 'output.csv'
    test_preds.to_csv(output_file, index=False)