
import numpy as np
import torch
import torch.nn as nn

def edge_pad(x):
    "Add zeros to right and bottom edges if dims are odd"
    *o, h, w = list(x.shape)
    h_pad = h%2 == 1
    if h_pad:
        x = torch.cat((x, torch.zeros(*o, 1, w)), dim=2)
        
    *o, h, w = list(x.shape)
    w_pad = w%2 == 1
    if w_pad:
        x = torch.cat((x, torch.zeros(*o, h, 1)), dim=3)
        
    return x, (h_pad, w_pad)


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
    def __init__(self, in_channels, bias=False):
        super().__init__()
        
        channel_scale = in_channels*4
        
        self.swish = Swish()
        
        self.feature_map = nn.Conv2d( in_channels=in_channels 
                                    , out_channels=channel_scale 
                                    , kernel_size=2 
                                    , stride=2
                                    , padding=0 
                                    , bias=bias
                                    ) 
        
        self.gn = nn.GroupNorm(num_groups=pick_n_groups(channel_scale), num_channels=channel_scale)
    
        self._init_wts()
   
    def _init_wts(self): 
        self.feature_map.weight.data = nn.init.kaiming_normal_(self.feature_map.weight.data)
    
    def forward(self, x, beta=1.0):
        f = self.feature_map(x)
        f = self.gn(f)
        f = self.swish(f)
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
        self._init_wts() 
     
    def _init_wts(self): 
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
    def __init__(self, in_channels, n_feature_maps, kernel_size, dropout=0, bias=False): 
        """ 
        PARSENET: LOOKING WIDER TO SEE BETTER: https://arxiv.org/pdf/1506.04579.pdf
        Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition: https://arxiv.org/pdf/1406.4729.pdf
        Rethinking Atrous Convolution for Semantic Image Segmentation: https://arxiv.org/pdf/1706.05587.pdf
        
        """ 
        super().__init__() 
         
        assert in_channels%2 == 0, 'in_channels must be even' 
        assert n_feature_maps%2 == 0, 'n_feature_maps must be even' 
                
        self.swish = Swish() 
         
        self.d_out = nn.Dropout2d(p=dropout) 
         
        self.gn1 = nn.GroupNorm(num_groups=pick_n_groups(in_channels), num_channels=in_channels) 
         
        self.bottleneck = nn.Conv2d( in_channels=in_channels 
                                   , out_channels=n_feature_maps*4 
                                   , kernel_size=1 
                                   , bias=bias
                                   ) 
         
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=kernel_size) 
        
        self.gn2 = nn.GroupNorm(num_groups=pick_n_groups(n_feature_maps*4), num_channels=n_feature_maps*4) 
        
        self.feature_map = nn.Conv2d( in_channels=n_feature_maps*4
                                    , out_channels=n_feature_maps 
                                    , kernel_size=kernel_size 
                                    , bias=bias
                                    ) 
             
        self._init_wts() 
     
    def _init_wts(self): 
        self.bottleneck.weight.data = nn.init.kaiming_normal_(self.bottleneck.weight.data)
        self.feature_map.weight.data = nn.init.kaiming_normal_(self.feature_map.weight.data) 
    
    def _forward(self, x, beta=1.0):
        xn = self.gn1(x) 
        f = self.d_out(xn) 
        f = self.swish(f, beta) 
        f = self.bottleneck(f) 
        f = self.adaptive_avg_pool(f)
        f = self.gn2(f) 
        f = self.swish(f, beta)
        f = self.feature_map(f)
        return f, xn 
    
    def forward(self, x, beta=1.0): 
        b, c, h, w = list(x.shape)
        
        gfv, xn = self._forward(x, beta)
         
        # Full feature stack: concat on channel dim 
        f = gfv.repeat(1,1,h,w)  # number of times to repeat for each dim (1x, 1x, Hx, Wx)
        f = torch.cat((xn, f), dim=1) 
        
        # Global feature vector
        gfv = gfv.reshape(b, -1)
        return f, gfv


class ReUnit(nn.Module):
    def __init__(self, in_channels, n_feature_maps, direction='vertical', bidirectional=True, dropout=0):
        """
        Args:
            in_channels (int)
            n_feature_maps (int)
            direction (str): 'vertical', 'horizontal'
            bidirectional (bool)
        """ 
        super().__init__()
        self.n_feature_maps = n_feature_maps
        
        if direction == 'horizontal':
            self.batch_first = True  # (batch, seq_len, input_size); row = batch
        else:
            self.batch_first = False  # (seq_len, batch, input_size); cols = batch
        
        self.bidirectional = bidirectional
        self.swish = Swish() 
         
        self.d_out = nn.Dropout2d(p=dropout) 
         
        self.gn1 = nn.GroupNorm(num_groups=pick_n_groups(in_channels), num_channels=in_channels) 
         
        self.bottleneck = nn.Conv2d( in_channels=in_channels 
                                   , out_channels=n_feature_maps*4 
                                   , kernel_size=1 
                                   , bias=False
                                   ) 
         
        self.gn2 = nn.GroupNorm(num_groups=pick_n_groups(n_feature_maps*4), num_channels=n_feature_maps*4)
        
        if bidirectional:
            hidden_size = int(n_feature_maps/2)
        else:
            hidden_size = n_feature_maps
        self.feature_map = nn.GRU(input_size=n_feature_maps*4
                                 , hidden_size=hidden_size
                                 , num_layers=1
                                 , batch_first=self.batch_first
                                 , dropout=dropout
                                 , bidirectional=bidirectional
                                 )
        self._init_wts()
    
    def _init_wts(self): 
        self.bottleneck.weight.data = nn.init.kaiming_normal_(self.bottleneck.weight.data)
        
        for name, param in self.feature_map.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, beta=1.0): 
        
        f = self.d_out(x) 
        f = self.gn1(f) 
        f = self.swish(f, beta) 
        f = self.bottleneck(f) 
        f = self.gn2(f)

        # Reshape for recurrent pass
        b, c, h, w = list(f.shape)
        if self.batch_first:
            # Batch = batch * rows
            f = f.permute(0, 2, 3, 1).contiguous()
            f = f.reshape(b*h, w, c)
        else:
            f = f.permute(2, 0, 3, 1).contiguous()
            f = f.reshape(h, b*w, c)
        f, _ = self.feature_map(f)

        # Reshape back
        c = self.n_feature_maps
        
        if self.batch_first:
            f = f.reshape(b, h, w, c)
            f = f.permute(0, 3, 1, 2).contiguous()
        else:
            f = f.reshape(h, b, w, c)
            f = f.permute(1, 3, 0, 2).contiguous()
        
        # Concat on channel dim 
        f = torch.cat((x, f), dim=1) 
        
        return f


class UpsampleUnit(nn.Module):
    """Depthwise-Separable Upsampling 
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, kernels_per_layer=4, dropout=0, bias=False):
        super().__init__()
        
        self.swish = Swish()
        
        self.d_out = nn.Dropout2d(p=dropout)
        
        self.gn1 = nn.GroupNorm(num_groups=pick_n_groups(in_channels), num_channels=in_channels) 
        self.gn2 = nn.GroupNorm(num_groups=pick_n_groups(in_channels*kernels_per_layer), num_channels=in_channels*kernels_per_layer)
        
        self.depthwise = nn.ConvTranspose2d(in_channels=in_channels
                                            , out_channels=in_channels*kernels_per_layer
                                            , kernel_size=kernel_size
                                            , stride=stride
                                            , groups=in_channels
                                           )
        
        self.pointwise = nn.Conv2d(in_channels*kernels_per_layer, out_channels, kernel_size=1)
        
        self._init_wts()
    
    def _init_wts(self): 
        self.depthwise.weight.data = nn.init.kaiming_normal_(self.depthwise.weight.data) 
        self.pointwise.weight.data = nn.init.kaiming_normal_(self.pointwise.weight.data) 
    
    def forward(self, x, h_pad=False, w_pad=False, beta=1):
        f = self.d_out(x) 
        f = self.gn1(f) 
        f = self.swish(f, beta) 
        f = self.depthwise(f)
        f = self.gn2(f) 
        f = self.swish(f, beta) 
        f = self.pointwise(f)
        
        if h_pad:
            f = f[:, :, :-1, :]
        if w_pad:
            f = f[:, :, :, :-1]
        
        return f


class DownsampleBlock(nn.Module):
    def __init__(self 
                 , in_channels 
                 , n_downsamples
                ): 
        super().__init__() 

        units = []
        for _ in range(n_downsamples):
            units.append(DownsampleUnit(in_channels)) 
            in_channels *= 4

        self.units = nn.ModuleList(units)
        self.out_channels = in_channels

    def forward(self, x):
        x, (h_pad, w_pad) = edge_pad(x) 
        
        for unit in self.units: 
            x = unit(x) 
             
        return x, (h_pad, w_pad)


class DenseBlock(nn.Module): 
    def __init__(self 
                 , in_channels 
                 , n_feature_maps
                 , block_size  
                 , kernel_sizes 
                 , swish_beta_scale=1 
                 , dilation=1 
                 , interleave_dilation=False 
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


class ReBlock(nn.Module):
    """
    ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation: https://arxiv.org/pdf/1511.07053.pdf
    """ 
    def __init__(self 
                , in_channels 
                , n_feature_maps
                , block_size  
                , bidirectional=True
                , swish_beta_scale=1 
                , dropout=0 
                ): 
        super().__init__() 
         
        self.swish_beta_scale = swish_beta_scale 
         
        directions = ['vertical', 'horizontal']
        
        units = [] 
        for i in range(block_size): 
            direction = directions[i%2]
            units.append(ReUnit(in_channels, n_feature_maps, direction, bidirectional, dropout))
            in_channels += n_feature_maps
             
        self.units = nn.ModuleList(units) 
        self.out_channels = in_channels 
         
    def forward(self, x, beta=1.0): 
        f = x
        for unit in self.units: 
            f = unit(f, self.swish_beta_scale*beta) 
             
        return f


class DenseDilationNet(nn.Module):
    """
    """ 
    def __init__(self 
                 , in_channels 
                 , n_feature_maps
                 , block_size
                 , n_blocks
                 , kernel_sizes
                 , global_kernel_size  
                 , dilation_seq=None
                 , dropout=0 
                ): 
        super().__init__() 

        if dilation_seq is None:
            dilation_seq = 2**np.arange(n_blocks)

        # Dense block sequence
        blocks = [] 
        for d in dilation_seq: 
            block = DenseBlock( in_channels 
                              , n_feature_maps
                              , block_size 
                              , kernel_sizes  
                              , dilation=d
                              , dropout=dropout 
                              ) 
            blocks.append(block)
            in_channels = block.out_channels
            
            if d > 1:
                # Recurrent block
                block = ReBlock( in_channels 
                               , n_feature_maps
                               , block_size=2
                               , dropout=dropout
                               )
                blocks.append(block)
                in_channels = block.out_channels
            

        # Add global context unit to end of dense stack
        self.n_global_features = n_feature_maps*block_size
        self.out_channels = in_channels + self.n_global_features
        gcu = GlobalContextUnit(in_channels, n_feature_maps*block_size, global_kernel_size, dropout)
        blocks.append(gcu)
    
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, beta=1.0): 
        f = x 
        for block in self.blocks[:-1]: 
            f = block(f, beta) 
        
        # feature stack, global feature vector
        f, gfv = self.blocks[-1](f, beta)
        
        return f, gfv


class PixelScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0):
        super().__init__()
         
        self.d_out = nn.Dropout2d(p=dropout) 

        self.swish = Swish()

        self.bottle_channels = int(np.ceil(in_channels/4)//2*2)
        self.bottleneck = nn.Conv2d( in_channels=in_channels
                                   , out_channels=self.bottle_channels
                                   , kernel_size=1
                                   , bias=False 
                                   ) 
        
        self.gn = nn.GroupNorm(num_groups=pick_n_groups(self.bottle_channels), num_channels=self.bottle_channels)

        self.scorer = nn.Conv2d( in_channels=self.bottle_channels
                               , out_channels=out_channels
                               , kernel_size=kernel_size
                               , padding=int(kernel_size//2)
                               , bias=True
                               )

        self._init_wts()

    def _init_wts(self): 
        self.bottleneck.weight.data = nn.init.kaiming_normal_(self.bottleneck.weight.data) 
        self.scorer.weight.data *= (6.0/(self.bottle_channels + 1))**0.5
        self.scorer.bias.data *= 0.0

    def forward(self, x, beta=1.0):
        f = self.d_out(x) 
        f = self.bottleneck(f)
        f = self.gn(f) 
        f = self.swish(f, beta) 
        f = self.d_out(f)
        f = self.scorer(f)

        return f


class WheatHeadDetector(nn.Module):
    """
    """ 
    def __init__(self 
                 , n_feature_maps=8
                 , block_size=4
                 , n_blocks=3
                 , dropout=0 
                ): 
        super().__init__() 

        vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
        vgg.eval()
        vgg_modules = list(vgg.features.children())
        self.vgg_featurizer = nn.Sequential(*vgg_modules)[:12]
        
        self.dense_net = DenseDilationNet(in_channels=128
                                           , n_feature_maps=n_feature_maps
                                           , block_size=block_size
                                           , n_blocks=n_blocks
                                           , kernel_sizes=[(3,3)]
                                           , global_kernel_size=3
                                           , dropout=dropout
                                           )
        
        self.num_bboxes_regressor = nn.Sequential(
              nn.Linear(self.dense_net.n_global_features, out_features=int(self.dense_net.n_global_features//2))
            , nn.ReLU()
            , nn.Linear(int(self.dense_net.n_global_features//2), out_features=1)
        )
        
        self.bbox_spread_regressor = PixelScorer(in_channels=self.dense_net.out_channels, out_channels=1, kernel_size=9)
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size=(8,8))
        
        self.segmentation_scorer = PixelScorer(in_channels=self.dense_net.out_channels, out_channels=1, kernel_size=9)
        
        self.bbox_scorer = PixelScorer(in_channels=self.dense_net.out_channels, out_channels=3, kernel_size=9)

        self._init_wts()

    def _init_wts(self):
        for m in self.num_bboxes_regressor:
            if isinstance(m, nn.Linear):
                m.weight = nn.init.normal_(m.weight, mean=0, std=(2/m.in_features)**0.5)
                m.bias = nn.init.constant_(m.bias, 0)

    def _featurize(self, x):
        with torch.no_grad():
            f = self.vgg_featurizer(x)
        
        f, gfv = self.dense_net(f)

        return f, gfv

    def _forward_train(self, x):
        f, gfv = self. _featurize(x)
        
        yh_n_bboxes = self.num_bboxes_regressor(gfv)
        
        yh_bbox_spread = self.bbox_spread_regressor(f)
        yh_bbox_spread = self.adaptive_pool(yh_bbox_spread)
        
        yh_seg = self.segmentation_scorer(f)
        
        yh_bboxes = self.bbox_scorer(f)

        return yh_n_bboxes, yh_bbox_spread, yh_seg, yh_bboxes

    @torch.no_grad()
    def _forward_inference(self, x):
        return self._forward_train(x)
        
    def forward(self, x):
        pass 