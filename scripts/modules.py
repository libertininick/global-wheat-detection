
import torch
import torch.nn as nn


def pick_n_groups(n_channels):
    """Calculates number of groups based on n_channels
    Referenced from Table 3b in Group Normalization
    https://arxiv.org/pdf/1803.08494.pdf
    """
    assert n_channels%2 == 0, 'n_channels must be even'
    
    if n_channels%16 == 0:
        n_groups = int(n_channels//16)
    elif n_channels%8 == 0:
        n_groups = int(n_channels//8)
    elif n_channels%4 == 0:
        n_groups = int(n_channels//4)
    else:
        n_groups = int(n_channels//2)

    return n_groups


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, beta=1.0):
        return x*self.sigmoid(beta*x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels, dropout=0, bias=False):
        super().__init__()

        self.swish = Swish() 
         
        self.d_out = nn.Dropout2d(p=dropout) 

        self.feature_map = nn.Conv2d( in_channels=in_channels
                                    , out_channels=4*in_channels # Keep number of total cells the same
                                    , kernel_size=2
                                    , stride=2 
                                    , padding=0 
                                    , bias=bias
                                    ) 
        self.gn = nn.GroupNorm(num_groups=pick_n_groups(4*in_channels), num_channels=out_channels)

        self._init()
    
    def _init(self): 
        self.feature_map.weight.data = nn.init.kaiming_normal_(self.feature_map.weight.data)
    
    def forward(self, x, beta=1.0): 
        f = self.d_out(x) 
        f = self.feature_map(f) 
        f = self.gn(f) 
        f = self.swish(f, beta) 

        return f


class DenseUnit(nn.Module): 
    def __init__(self, in_channels, n_feature_maps, kernel_size, dilation=1, dropout=0, bias=False): 
        """ 
        Dilation = [1,2,4,8,16] 
        bias=False per https://arxiv.org/pdf/1606.02147.pdf
        """ 
        super().__init__() 
         
        assert in_channels%2 == 0, 'in_channels must be even' 
        assert n_feature_maps%2 == 0, 'n_feature_maps must be even' 
        assert dilation in [1,2,4,8,16], 'dilation must be one of [1,2,4,8,16]' 
         
        h, w = kernel_size 
         
        self.swish = Swish() 
         
        self.d_out = nn.Dropout2d(p=dropout) 
         
        self.gn1 = nn.GroupNorm(num_groups=pick_n_groups(in_channels), num_channels=in_channels) 
         
        self.bottleneck = nn.Conv2d( in_channels=in_channels 
                                   , out_channels=n_feature_maps*4 
                                   , kernel_size=1 
                                   , bias=bias
                                   ) 
         
        self.gn2 = nn.GroupNorm(num_groups=pick_n_groups(n_feature_maps*4), num_channels=n_feature_maps*4) 
         
        self.feature_map = nn.Conv2d( in_channels=n_feature_maps*4 
                                    , out_channels=n_feature_maps 
                                    , kernel_size=kernel_size 
                                    , padding=(int(h//2*dilation), int(w//2*dilation)) 
                                    , dilation=dilation 
                                    , bias=bias
                                    ) 
        self._init() 
     
    def _init(self): 
        self.bottleneck.weight.data = nn.init.kaiming_normal_(self.bottleneck.weight.data) 
        self.feature_map.weight.data = nn.init.kaiming_normal_(self.feature_map.weight.data) 
         
    def forward(self, x, beta=1.0): 
        f = self.d_out(x) 
        f = self.gn1(f) 
        f = self.swish(f, beta) 
        f = self.bottleneck(f) 
        f = self.gn2(f) 
        f = self.swish(f, beta)  
        f = self.feature_map(f) 
         
        # Concat on channel dim 
        f = torch.cat((x, f), dim=1) 
         
        return f


class GlobalContextUnit(nn.Module): 
    def __init__(self, in_channels, n_feature_maps, out_channels, kernel_size, dropout=0, bias=False): 
        """ 
        Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition: https://arxiv.org/pdf/1406.4729.pdf
        Rethinking Atrous Convolution for Semantic Image Segmentation: https://arxiv.org/pdf/1706.05587.pdf
        
        """ 
        super().__init__() 
         
        assert in_channels%2 == 0, 'in_channels must be even' 
        assert n_feature_maps%2 == 0, 'n_feature_maps must be even' 
        assert out_channels%2 == 0, 'out_channels must be even' 
                
        self.swish = Swish() 
         
        self.d_out = nn.Dropout2d(p=dropout) 
         
        self.gn1 = nn.GroupNorm(num_groups=pick_n_groups(in_channels), num_channels=in_channels) 
         
        self.bottleneck1 = nn.Conv2d( in_channels=in_channels 
                                   , out_channels=n_feature_maps*4 
                                   , kernel_size=1 
                                   , bias=bias
                                   ) 
         
        self.gn2 = nn.GroupNorm(num_groups=pick_n_groups(n_feature_maps*4), num_channels=n_feature_maps*4) 
        self.bottleneck2 = nn.Conv2d( in_channels=n_feature_maps*4 
                                   , out_channels=n_feature_maps 
                                   , kernel_size=1 
                                   , bias=bias
                                   ) 
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=kernel_size) 
        
        self.gn3 = nn.GroupNorm(num_groups=pick_n_groups(n_feature_maps), num_channels=n_feature_maps)
        self.feature_map = nn.Conv2d( in_channels=n_feature_maps
                                    , out_channels=out_channels 
                                    , kernel_size=kernel_size 
                                    , bias=bias
                                    ) 
             
        self._init() 
     
    def _init(self): 
        self.bottleneck1.weight.data = nn.init.kaiming_normal_(self.bottleneck1.weight.data)
        self.bottleneck2.weight.data = nn.init.kaiming_normal_(self.bottleneck2.weight.data)  
        self.feature_map.weight.data = nn.init.kaiming_normal_(self.feature_map.weight.data) 
    
    def _forward(self, x, beta=1.0):
        xn = self.gn1(x)
        f = self.d_out(xn) 
        f = self.swish(f, beta) 
        f = self.bottleneck1(f) 
        f = self.gn2(f) 
        f = self.swish(f, beta) 
        f = self.bottleneck2(f) 
        f = self.adaptive_avg_pool(f)
        f = self.gn3(f) 
        f = self.swish(f, beta)
        f = self.feature_map(f)
        return f, xn 
    
    def forward(self, x, beta=1.0): 
        f, xn = self._forward(x, beta)
        
        # Concat on channel dim 
        f = f.repeat(1,1,*list(x.shape)[-2:])
        f = torch.cat((xn, f), dim=1) 
         
        return f
    
    def feature_vector(self, x, beta=1.0):
        """Returns a global feature vector for the input
        """
        f = self._forward(x, beta)
        
        return f.squeeze(-1).squeeze(-1)


class DenseBlock(nn.Module): 
    def __init__(self 
                 , block_size 
                 , in_channels 
                 , n_feature_maps 
                 , kernel_sizes 
                 , swish_beta_scale=1 
                 , dilation=1 
                 , interleave_dilation=True 
                 , dropout=0 
                ): 
        super().__init__() 
         
        self.swish_beta_scale = swish_beta_scale 
         
        n_kernel_sizes = len(kernel_sizes) 
        dilate = dilation > 1 
         
        units = [] 
        for i in range(block_size): 
            kernel_size = kernel_sizes[i%n_kernel_sizes] 
             
            if dilate: 
                d = dilation 
            else: 
                d = 1 
             
            units.append(DenseUnit(in_channels, n_feature_maps, kernel_size, d, dropout)) 
             
            if ((i+1)%n_kernel_sizes == 0) and interleave_dilation: 
                dilate = not dilate 
             
            in_channels += n_feature_maps 
             
        self.units = nn.ModuleList(units) 
        self.out_channels = in_channels 
         
    def forward(self, x, beta=1.0): 
        f = x 
        for unit in self.units: 
            f = unit(f, self.swish_beta_scale*beta) 
             
        return f


class DenseDilationNet(nn.Module): 
    def __init__(self 
                 , block_size 
                 , in_channels 
                 , n_feature_maps
                 , kernel_sizes
                 , n_global_feats
                 , global_kernel_size  
                 , swish_beta_scales
                 , dilations 
                 , interleave_dilation=True 
                 , dropout=0 
                ): 
        super().__init__() 
         
        assert len(swish_beta_scales) == len(dilations), 'Lengths must match' 
         
        blocks = [] 
        for sbs, d in zip(swish_beta_scales, dilations): 
            block = DenseBlock( block_size 
                              , in_channels 
                              , n_feature_maps 
                              , kernel_sizes 
                              , swish_beta_scale=sbs 
                              , dilation=d 
                              , interleave_dilation=interleave_dilation 
                              , dropout=dropout 
                              ) 
            blocks.append(block) 
            in_channels = block.out_channels 
        
        # Add global context unit to end of dense stack
        gcu = GlobalContextUnit(in_channels, n_feature_maps, n_global_feats, global_kernel_size, dropout)
        blocks.append(gcu)
        self.out_channels = in_channels + n_global_feats
             
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, beta=1.0): 
        f = x 
        for block in self.blocks: 
            f = block(f, beta) 
             
        return f
    
    def feature_vector(self, x, beta=1.0):
        """Returns a global feature vector for the input
        """
        f = x 
        for block in self.blocks[:-1]: 
            f = block(f, beta) 
        f = self.blocks[-1].feature_vector(f, beta)
        
        return f