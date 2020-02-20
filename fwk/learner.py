import torch
import sklearn
import torch.nn.functional as F


def train(args, model, device, train_loader, optimizer, epoch):
    torch.cuda.synchronize()
    model.train()
    for batch_idx, (data_t, target_t) in enumerate(train_loader):
        data = data_t.to(device)
        target = target_t.to(device)
        optimizer.zero_grad()
        loss = sum([F.nll_loss(model(data[i, :]).unsqueeze(0), torch.argmax(target[i, :]).unsqueeze(-1)) for i in
                    range(data.shape[0])])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:5d}/{:5d} ({:2.0f}%)] Loss: {:1.5e}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    torch.cuda.synchronize()


def test(_, model, device, test_loader, epoch):
    model.eval()
    sum_correct = 0.
    test_loss = 0.
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data_t, target_t in test_loader:
            data = data_t.to(device)
            target = target_t.to(device)
            N = data.shape[0]
            outputs = [model(data[i, :]) for i in range(N)]
            preds = [outputs[i].argmax() for i in range(N)]
            targets = [target[i, :].argmax() for i in range(N)]
            all_targets.extend([targets[i].item() for i in targets])
            all_preds.extend([preds[i].item() for i in preds])
            correct = sum([targets[i] == preds[i] for i in range(N)]).item()
            sum_correct += correct
            # print(float(correct)/N)
            test_loss += sum([F.nll_loss(outputs[i].unsqueeze(0), targets[i].unsqueeze(-1)) for i in range(N)])

    test_loss /= len(test_loader.dataset)

    # print('')
    print('Epoch: {:3d}, AvgLoss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch, test_loss, float(sum_correct) / len(test_loader.dataset)))
    # print('')

    print('')
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(all_targets, all_preds))
    print('')
    print('Epoch: {:3d}, AvgLoss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch, test_loss, float(sum_correct) / len(test_loader.dataset)))
    print('')
