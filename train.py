import torch
import torch.optim as optim
import torch.nn.functional as F
import utils.dataset
import utils.profile
from torch.utils.data import DataLoader
import tools

crop_size = (640, 1024)
crop_seed = None
max_version = 50
device = 'cuda'
version = None
seed = 0
batch = 2

profile = utils.profile.GreenNet()
model = profile.load_model(version=version)[1].to(device)

version, loss_history = profile.load_history(version)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

dataset = utils.dataset.瑞評大帥哥幫我train資料集(crop_size=crop_size, crop_seed=crop_seed)

train_dataset, test_dataset = utils.dataset.random_split(dataset, seed=seed, train_ratio=0.9)

# train_dataset = utils.dataset.random_subset(train_dataset, 4)
# test_dataset = utils.dataset.random_subset(test_dataset, 4)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

print('Number of training data:', len(train_dataset))
print('Number of testing data:', len(test_dataset))
print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', dataset)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

for v in range(version, max_version + 1):
    train_loss = []
    test_loss = []

    print('Start training, version = {}'.format(v))
    model.train()
    for batch_index, (X, Y) in enumerate(train_loader):
        tools.tic()
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        predict_ndvi = model.forward(X)

        mse_loss = F.mse_loss(predict_ndvi, Y)
        mse_loss.backward()
        optimizer.step()

        train_loss.append(float(mse_loss))
        time = utils.timespan_str(tools.toc(True))
        print(f'[{v}/{max_version} - {batch_index + 1}/{len(train_loader)} {time}] loss = {mse_loss: .3f}')

    train_loss = float(torch.tensor(train_loss).mean())

    print('Start testing, version = {}'.format(v))
    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        tools.tic()
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            predict_ndvi = model.forward(X)
            mse_loss = F.mse_loss(predict_ndvi, Y)
            test_loss.append(float(mse_loss))
            time = utils.timespan_str(tools.toc(True))
            print(f'[{v}/{max_version} - {batch_index + 1}/{len(test_loader)} {time}] val_loss = {mse_loss: .3f}')

    test_loss = float(torch.tensor(test_loss).mean())
    print(f'Avg val_loss = {test_loss:.3f}')

    loss_history['train'].append(train_loss)
    loss_history['test'].append(test_loss)

    print('Start save model')
    profile.save_version(model, loss_history, v)