import torch
import torch.nn as nn

class Unet(nn.Module):
    '''
    Unet model as proposed by paper 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox
    code courtesy youtube video Abhishek Thakur
    '''
    def __init__(self):
        super(Unet,self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv1 = self.double_conv(1,64)
        self.down_conv2 = self.double_conv(64,128)
        self.down_conv3 = self.double_conv(128,256)
        self.down_conv4 = self.double_conv(256,512)
        self.down_conv5 = self.double_conv(512,1024)
        
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.up_conv1 = self.double_conv(1024,512)
        
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.up_conv2 = self.double_conv(512,256)
        
        self.conv_trans3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.up_conv3 = self.double_conv(256,128)
        
        self.conv_trans4 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.up_conv4 = self.double_conv(128,64)
        
        self.out = nn.Conv2d(
            in_channels = 64,
            out_channels = 2,
            kernel_size = 1
        )


    def double_conv(in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=3),
            nn.ReLU(inplace=True),
        )
    
    def crop_tensor(tensor, target_tensor):
        '''
        Adjusting the shapes of layers to concatenate
        
        '''
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        # Assuming tensor size is larger than target size
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]
    
    def forward(self,x):
        #contract path 
        x1 = self.down_conv1(1,64)(x)
        x2 = self.max_pool_2x2(x1)
        
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)

        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)

        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)

        x9 = self.down_conv5(x8)

        #expand path 

        x= self.conv_trans1(x9)
        y = self.crop_tensor(x7,x)
        x = self.up_conv1(torch.cat([x,y],1))

        x= self.conv_trans2(x)
        y = self.crop_tensor(x5,x)
        x = self.up_conv2(torch.cat([x,y],1))

        x= self.conv_trans3(x)
        y = self.crop_tensor(x3,x)
        x = self.up_conv3(torch.cat([x,y],1))

        x= self.conv_trans4(x)
        y = self.crop_tensor(x1,x)
        x = self.up_conv4(torch.cat([x,y],1))

        x = self.out(x)

        return x

        


        


