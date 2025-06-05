# encoding: utf-8

import functools

import torch
import torch.nn as nn


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet.  下采样的次数。For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
#下面的参数input_nc就是输入通道数，
# if DDH, input_nc=opt.channel_secret*opt.num_secret+opt.channel_cover*opt.num_cover;
# if UDH, input_nc=opt.channel_secret*opt.num_secret, 
#output_nv是输出通道数
# output_nc=opt.channel_cover*opt.num_cover.

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
        self.tanh = output_function==nn.Tanh
        if self.tanh:
            self.factor = 10/255
        else:
            self.factor = 1.0

    def forward(self, input):
        return self.factor*self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
    # outer_nc and inner_nc: 输入、输出的通道数
    # submodule: 指向下一个更深的UnetSkipConnectionBlock，递归地堆叠U-Net层。
    # outermost and innermost: 指定该块是网络中的最外层还是最内层。
    # use_dropout: 如果为True，则为块添加dropout层。
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        #下采样
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        #Skip Connection：如果outermost为False，它将输入与上采样路径的输出连接起来。
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upconv]
            else: 
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        
if __name__ == '__main__':
    import time
    device = 'cuda'
    encoder = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, norm_layer=nn.BatchNorm2d, output_function=nn.Sigmoid)#.cuda()
    H_input = torch.rand([8,3,128,128])#.cuda()
    start = time.time()
    container = encoder(H_input)
    end = time.time()
    print (end - start)
    # print(container.shape)
    # print(str(encoder))
    
    #参数计算
    # from calflops import calculate_flops# from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # input_shape = (1,3,128,128)
    # flops, macs, params = calculate_flops(model=encoder,input_shape=input_shape)
    #                                   # output_as_string=True,
    #                                   # output_precision=4)
    # print("HidingUNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    
