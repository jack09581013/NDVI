import torch
import torch.optim as optim
import torch.nn.functional as F
import utils.dataset
import utils.profile
from torch.utils.data import DataLoader
import tools

crop_size = (896, 1216)
# crop_size = (448, 608)
crop_seed = None
max_version = 50
device = 'cuda'
version = None
seed = 0
batch = 1

profile = utils.profile.GreenNet()
model = profile.load_model(version=version)[1].to(device)

dataset = utils.dataset.瑞評大帥哥幫我train資料集(crop_size=crop_size, crop_seed=crop_seed)

test_dataset = utils.dataset.random_split(dataset, seed=seed, train_ratio=0.9)[1]

# test_dataset = utils.dataset.random_subset(test_dataset, 4)

test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

print('Number of testing data:', len(test_dataset))
print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', dataset)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

test_loss = []

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
        print(f'[{batch_index + 1}/{len(test_loader)} {time}] val_loss = {mse_loss: .3f}')

        # if mse_loss > 0:
        #     utils.plot_image_disparity(X[0], Y[0, 0], predict_ndvi[0, 0], float(mse_loss),
        #                                save_file=f'result/{batch_index}.png')

test_loss = float(torch.tensor(test_loss).mean().sqrt())
print(f'RMSE = {test_loss:.3f}')
