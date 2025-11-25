from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')
from model_cifar import UNet
import torch
from trainer import ModelTrainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from normalizer import DataNormalizer
train_data,test_data=load_q2_data()
train_data=train_data.data
test_data=test_data.data
train_normalizer=DataNormalizer(data_min=np.min(train_data),data_max=np.max(train_data))
test_normalizer=DataNormalizer(data_min=np.min(test_data),data_max=np.max(test_data))
print("Train Normalizer Parameters")
print(train_normalizer.data_min,train_normalizer.data_max)
print("Test Normalizer Parameters")
print(test_normalizer.data_min,test_normalizer.data_max)
train_data_normalized=train_normalizer.normalize(train_data)
test_data_normalized=test_normalizer.normalize(test_data)
print(train_data_normalized.shape,test_data_normalized.shape)
print(train_data_normalized.min(),train_data_normalized.max())
print(test_data_normalized.min(),test_data_normalized.max())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
train_data_torch = torch.from_numpy(train_data_normalized).float()
test_data_torch = torch.from_numpy(test_data_normalized).float()
train_data_torch=train_data_torch.permute(0,3,1,2)
test_data_torch=test_data_torch.permute(0,3,1,2)
    
    # Create model and trainer
model = UNet(
    in_channels=3,  # For RGB images
    hidden_dims=[64, 128, 256, 512],  # As specified in the homework
    blocks_per_dim=2  # As specified in the homework
)

trainer = ModelTrainer(model, device, train_data_torch, test_data_torch, 
                           batch_size=256, lr=1e-3,warmup_steps=100,num_epochs=60)
    
    # Train the model
trainer.train( print_every=10)
    
    # Plot losses
trainer.plot_losses()
trainer.save_model('model_test.pth')