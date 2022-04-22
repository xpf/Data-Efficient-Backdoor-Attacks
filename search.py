import tqdm, torch, os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from opts import get_opts
from utils.utils import get_name
from utils.settings import DATASETTINGS
from models import build_model
from datasets import build_transform, build_data
from attacks import build_trigger


def search(opts):
    name = get_name(opts, 'search')
    print('search', name)
    DSET = DATASETTINGS[opts.data_name]
    train_transform = build_transform(True, DSET['img_size'], DSET['crop'], DSET['flip'])
    val_transform = build_transform(False, DSET['img_size'], DSET['crop'], DSET['flip'])
    trigger = build_trigger(opts.attack_name, DSET['img_size'], DSET['num_data'], mode=0, target=opts.target, trigger=opts.trigger)
    train_data = build_data(opts.data_name, opts.data_path, True, trigger, train_transform)
    val_data = build_data(opts.data_name, opts.data_path, False, trigger, val_transform)
    poison_num = int(len(train_data.targets) * opts.ratio)
    shuffle = np.arange(len(train_data.targets))[np.array(train_data.targets) != opts.target]  # select poisoned samples from data of non-target classes only
    np.random.shuffle(shuffle)
    samples_idx = shuffle[:poison_num]  # create random poison samples idx

    for n in range(opts.n_iter):
        print('searching with {:2d} iteration'.format(n))
        model = build_model(opts.model_name, DSET['num_classes']).to(opts.device)
        train_data = build_data(opts.data_name, opts.data_path, True, trigger, train_transform)
        train_data.data = np.concatenate((train_data.data, train_data.data[samples_idx]), axis=0)  # append selected poisoned samples to the clean train dataset
        train_data.targets = train_data.targets + [train_data.targets[i] for i in samples_idx]
        train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=2)
        val_loader = DataLoader(dataset=val_data, batch_size=256, shuffle=False, num_workers=2)

        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], 0.1)
        criterion = nn.CrossEntropyLoss().to(opts.device)

        correctness = []
        for epoch in range(70):
            trigger.set_mode(0), model.train()
            correct, total, ps, ds = 0, 0, [], []
            desc = 'train - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(train_loader, desc=desc.format(epoch, 0, 0, 0), disable=opts.disable)
            for x, y, b, s, d in run_tqdm:
                x, y, b, s, d = x.to(opts.device), y.to(opts.device), b.to(opts.device), s.to(opts.device), d.to(opts.device)
                optimizer.zero_grad()
                p = model(x)
                loss_cls = criterion(p, y)
                loss_cls.backward()
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]

                ps.append((p == y)[d >= DSET['num_data']].long().detach().cpu().numpy())  # record correctness of poisoned samples
                ds.append(d[d >= DSET['num_data']].detach().cpu().numpy())

                optimizer.step()
                run_tqdm.set_description(desc.format(epoch, correct / (total + 1e-12)))
            scheduler.step()
            train_acc = correct / (total + 1e-8)

            ps, ds = np.concatenate(ps, axis=0), np.concatenate(ds, axis=0)
            ps = ps[np.argsort(ds)]  # from small to large
            correctness.append(ps[:, np.newaxis])  # record correctness per epoch

            trigger.set_mode(1), model.eval()
            correct, total = 0, 0
            desc = 'val   - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable)
            for x, y, _, _, _ in run_tqdm:
                x, y = x.to(opts.device), y.to(opts.device)
                with torch.no_grad():
                    p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
                run_tqdm.set_description(desc.format(epoch, correct / total))
            val_acc = correct / (total + 1e-8)

            trigger.set_mode(2), model.eval()
            correct, total = 0, 0
            desc = 'back  - epoch: {:3d}, acc: {:.3f}'
            run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable)
            for x, y, b, _, _ in run_tqdm:
                x, y, b = x.to(opts.device), y.to(opts.device), b.to(opts.device)
                idx = b == 1
                x, y, b = x[idx, :, :, :], y[idx], b[idx]
                if x.shape[0] == 0: continue
                with torch.no_grad():
                    p = model(x)
                _, p = torch.max(p, dim=1)
                correct += (p == y).sum().item()
                total += y.shape[0]
                run_tqdm.set_description(desc.format(epoch, correct / total))
            back_acc = correct / (total + 1e-8)

            if opts.disable:
                print('epoch: {:3d}, train_acc: {:.3f}, val_acc: {:.3f}, back_acc: {:.3f}'.format(epoch, train_acc, val_acc, back_acc))

        correctness = np.concatenate(correctness, axis=1)
        diff = correctness[:, 1:] - correctness[:, :-1]
        forget_events = np.sum(diff == -1, axis=1)

        forget_events_idx = np.argsort(forget_events)
        samples_idx = samples_idx[forget_events_idx][::-1]  # sort the selected poisoned samples in order of FEs from large to small
        samples_idx = samples_idx[:int(len(samples_idx) * opts.alpha)]  # retain a certain number of poisoned samples

        np.random.shuffle(shuffle)
        samples_idx = np.concatenate((samples_idx, shuffle[:(poison_num - len(samples_idx))]), axis=0)  # random add new poisoned samples from the pool

    np.save(os.path.join(opts.sample_path, '{}.npy'.format(name)), samples_idx)  # save the selected poisoned samples


if __name__ == '__main__':
    opts = get_opts()
    search(opts)
