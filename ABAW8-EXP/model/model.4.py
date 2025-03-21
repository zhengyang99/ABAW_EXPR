from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet
import math
import os
import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, input_channels=3, num_classes=8, use_pretrained=True, state_dict_name='', root_dir='', mode="ir",
                 embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, num_layers=50, drop_ratio=0.4, mode=mode)
        if use_pretrained:
            path = os.path.join(root_dir, state_dict_name + ".pth")
            state_dict = torch.load(path, map_location='cpu')

            if "backbone" in list(state_dict.keys())[0]:

                self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                        Dropout(0.4),
                                                        Flatten(),
                                                        Linear(embedding_dim * 5 * 5, embedding_dim),
                                                        BatchNorm1d(embedding_dim))

                new_state_dict = {}
                for key, value in state_dict.items():

                    if "logits" not in key:
                        new_key = key[9:]
                        new_state_dict[new_key] = value

                self.backbone.load_state_dict(new_state_dict)
            else:
                self.backbone.load_state_dict(state_dict)

            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x


class my_2d1d(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
            state_dict = torch.load(path, map_location='cpu')
            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
            dropout=self.dropout)
        self.regressor = nn.Linear(self.embedding_dim // 4, self.output_dim)
        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


# class my_2d1ddy(nn.Module):
#     def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
#                  output_dim=1, kernel_size=5, dropout=0.1, root_dir='', input_dim_other=[128, 39]):
#         super().__init__()

#         self.modality = modality
#         self.backbone_state_dict = backbone_state_dict
#         self.backbone_mode = backbone_mode
#         self.root_dir = root_dir

#         self.embedding_dim = embedding_dim
#         self.channels = channels
#         self.output_dim = output_dim
#         self.kernel_size = kernel_size
#         self.attention = attention
#         self.dropout = dropout
#         self.other_feature_dim = input_dim_other  # input dimension of other modalities

#     def init(self, fold=None):
#         path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

#         spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
#         state_dict = torch.load(path, map_location='cpu')
#         spatial.load_state_dict(state_dict)

#         for param in spatial.parameters():
#             param.requires_grad = False


#         # for name,param in spatial.named_parameters():
#         #     if 'res_layer.4' in name:
#         #         param.requires_grad = True

#         # for param in spatial.parameters():
#         #     param.requires_grad = True

#         self.spatial = spatial.backbone

#         self.temporal = TemporalConvNet(
#             num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
#             dropout=self.dropout)

#         # self.temporal1 = TemporalConvNet(
#         #     num_inputs=self.other_feature_dim[0], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
#         #     attention=self.attention,
#         #     dropout=self.dropout)
#         # self.temporal2 = TemporalConvNet(
#         #     num_inputs=self.other_feature_dim[1], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
#         #     attention=self.attention,
#         #     dropout=self.dropout)
#         '''
#         self.temporal3 = TemporalConvNet(
#             num_inputs=self.other_feature_dim[2], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
#             attention=self.attention,
#             dropout=self.dropout)
#         self.temporal4 = TemporalConvNet(
#             num_inputs=self.other_feature_dim[3], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
#             attention=self.attention,
#             dropout=self.dropout)
#         '''

#         hidden_size =  [256,128,64]  
#         # self.feat_fc = nn.Conv1d(192, hidden_size[0], 1, padding=0)
#         self.feat_fc = nn.Conv1d(128, hidden_size[0], 1, padding=0)

#         self.transformer = TransEncoder(inc=hidden_size[0], outc=hidden_size[1], dropout=0.3, nheads=8, nlayer=4)
#         self.activ = nn.LeakyReLU(0.1) 
#         self.dropout = nn.Dropout(p=0.3)



#         # self.encoder1 = nn.Linear(self.embedding_dim // 4, 32) # 128,32
#         # self.encoder2 = nn.Linear(self.other_feature_dim[0] // 4, 32) #128,32
#         # self.encoder3 = nn.Linear(32, 32)

#         # self.encoderQ1 = nn.Linear(self.embedding_dim // 4, 32)# 128,32
#         # self.encoderQ2 = nn.Linear(self.other_feature_dim[0] // 4, 32) #128,32
#         # self.encoderQ3 = nn.Linear(32, 32)

#         # self.encoderV1 = nn.Linear(self.embedding_dim // 4, 32) # 128,32
#         # self.encoderV2 = nn.Linear(self.other_feature_dim[0] // 4, 32)  #128,32
#         # self.encoderV3 = nn.Linear(32, 32)
#         #self.encoder4 = nn.Linear(self.other_feature_dim[3] // 4, 32)
#         # self.gn1 = nn.GroupNorm(8, 32)
#         # self.gn2 = nn.GroupNorm(8, 32)

#         # self.ln = nn.LayerNorm([3, 32])

#         # self.AUhead = nn.Sequential(
#         #         nn.Linear(hidden_size[1], hidden_size[2]),
#         #         nn.BatchNorm1d(hidden_size[2]),
#         #         nn.Linear(hidden_size[2], 8),
#         #         )
#         # self.regressor = nn.Linear(224, self.output_dim)

#         self.regressor = nn.Linear(128, 8)

#         # self.regressor = Sequential(
#         #     BatchNorm1d(128),
#         #     Dropout(0.4),
#         #     Linear(128, 8)
#         # )

#     def forward(self, x, x1, x2):
#         num_batches, length, channel, width, height = x.shape
#         x = x.view(-1, channel, width, height)
#         x = self.spatial(x)
#         _, feature_dim = x.shape
#         x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()

#         # x = self.temporal(x)
#         x = self.temporal(x).contiguous()
#         print("x.shape", x.shape) # [16, 128, 300]
 

#         bs,_,seq_len=x.shape


#         # feat = torch.cat(tensors = list(X.values()),dim=1) # bs,224,win

#         # feat=torch.cat(tensors =[x,x1,x2],dim=1) #torch.Size([16, 192, 300])
#         # feat=torch.cat(tensors =[x,x,x],dim=1) #torch.Size([16, 192, 300])
#         feat=x

#         feat = self.feat_fc(feat)
#         feat = self.activ(feat)
#         out = self.transformer(feat)

#         out = torch.transpose(out, 1, 0)
#         out = torch.reshape(out, (bs*seq_len, -1))

#         print("out.shape", out.shape) # [4800, 128]
#         out = self.regressor(out)
#         print("out.shape", out.shape) # [4800, 8]

#         out = out.view(bs, seq_len, -1)
#         print("out.shape", out.shape) # [16, 300, 8]
        
#         return out


class my_2dlstm(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality='frame', embedding_dim=512, hidden_dim=256,
                 output_dim=1, dropout=0.5, root_dir=''):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        if 'frame' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        elif 'eeg_image' in self.modality:
            spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False, num_classes=3,
                               input_channels=6)

            if fold is not None:
                path = os.path.join(self.root_dir, self.backbone_state_dict + "_" + str(fold) + ".pth")

            state_dict = torch.load(path, map_location='cpu')

            spatial.load_state_dict(state_dict)
        else:
            raise ValueError("Unsupported modality!")

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=self.dropout)
        self.regressor = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).contiguous()
        x, _ = self.temporal(x)
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


class my_temporal(nn.Module):
    def __init__(self, model_name, num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5, cnn1d_dropout_rate=0.1,
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        if "1d" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                       kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2

            self.regressor = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        features = {}
        if "lstm_only" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()
        batch, time_step, temporal_feature_dim = x.shape

        x = x.view(-1, temporal_feature_dim)
        x = self.regressor(x).contiguous()
        x = x.view(batch, time_step, self.output_dim)
        return x




class TransEncoder(nn.Module):
    def __init__(self, inc=512, outc=512, dropout=0.6, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.d_model = outc
        self.dim_feedforward = outc
        self.dropout = dropout
        self.conv1 = nn.Conv1d(inc, self.d_model, kernel_size=1, stride=1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        return out


class my_model2(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir='', input_dim_other=[128, 39]):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout
        self.other_feature_dim = input_dim_other  # input dimension of other modalities

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
        state_dict = torch.load(path, map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=3, attention=self.attention,
            dropout=self.dropout)
        self.temporalf = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=5, attention=self.attention,
            dropout=self.dropout)
        self.temporalf1 = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=7, attention=self.attention,
            dropout=self.dropout)

        self.temporal1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=3,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal1v = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=5,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal1v1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[64, 64, 64, 64], kernel_size=7,
            attention=self.attention,
            dropout=self.dropout)

        self.temporal2 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=3,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2m = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=5,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2m1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[64, 64, 64, 64], kernel_size=7,
            attention=self.attention,
            dropout=self.dropout)
        '''
        self.temporal3 = TemporalConvNet(
            num_inputs=self.other_feature_dim[2], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal4 = TemporalConvNet(
            num_inputs=self.other_feature_dim[3], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        '''

        self.encoder1 = nn.Linear(self.embedding_dim // 4 * 3, 64)
        self.encoder2 = nn.Linear(64*3, 64)
        self.encoder3 = nn.Linear(64*3, 64)

        self.encoderQ1 = nn.Linear(self.embedding_dim // 4 * 3, 64)
        self.encoderQ2 = nn.Linear(64, 64)
        self.encoderQ3 = nn.Linear(64, 64)

        self.encoderV1 = nn.Linear(self.embedding_dim // 4 * 3, 64)
        self.encoderV2 = nn.Linear(64, 64)
        self.encoderV3 = nn.Linear(64, 64)
        #self.encoder4 = nn.Linear(self.other_feature_dim[3] // 4, 32)
        self.gn1 = nn.GroupNorm(8, 32)

        self.gn2 = nn.GroupNorm(8, 32)

        self.ln = nn.LayerNorm([3, 64])

        # self.regressor = nn.Linear(480, self.output_dim)
        self.regressor = nn.Sequential(
            nn.Linear(384+64*3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 8)
        )

    def forward(self, x, x1, x2):
        # print("mymodel2222222222222222")
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        
        xa = self.temporalf(x).transpose(1, 2).contiguous()
        xb = self.temporalf1(x).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)

        xa = xa.contiguous().view(num_batches * length, -1)
        xb = xb.contiguous().view(num_batches * length, -1)

        if len(x1) > 1 and len(x2) > 1:
            x1 = x1.squeeze().transpose(1, 2).contiguous().float()
            x2 = x2.squeeze().transpose(1, 2).contiguous().float()
        else:
            x1 = x1.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
            x2 = x2.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
        #x3 = x3.transpose(1, 2).contiguous().float()
        #x4 = x4.transpose(1, 2).contiguous().float()

        
        x1a = self.temporal1v(x1).transpose(1, 2).contiguous()
        x1b = self.temporal1v1(x1).transpose(1, 2).contiguous()
        x1 = self.temporal1(x1).transpose(1, 2).contiguous()
        
        x2a = self.temporal2m(x2).transpose(1, 2).contiguous()
        x2b = self.temporal2m1(x2).transpose(1, 2).contiguous()
        x2 = self.temporal2(x2).transpose(1, 2).contiguous()
        #x3 = self.temporal3(x3).transpose(1, 2).contiguous()
        #x4 = self.temporal4(x4).transpose(1, 2).contiguous()
        x = torch.cat([x, xa, xb], dim=-1) #[4800, 128*3]
        x1 = torch.cat([x1, x1a, x1b], dim=-1) #[16, 300, 64*3]
        x2 = torch.cat([x2, x2a, x2b], dim=-1) #[16, 300, 64*3]

        x0 = self.encoder1(x)
        x1 = self.encoder2(x1.contiguous().view(num_batches * length, -1))
        x2 = self.encoder3(x2.contiguous().view(num_batches * length, -1))
        # print("x.shape", x.shape) # [4800, 384]
        # print("x0.shape", x0.shape) # [4800, 64]
        # print("x1.shape", x1.shape) # [4800, 64]
        # print("x2.shape", x2.shape) # [4800, 64]

        xq0 = self.encoderQ1(x)
        xq1 = self.encoderQ2(x1.contiguous().view(num_batches * length, -1))
        xq2 = self.encoderQ3(x2.contiguous().view(num_batches * length, -1))

        xv0 = self.encoderV1(x)
        xv1 = self.encoderV2(x1.contiguous().view(num_batches * length, -1))
        xv2 = self.encoderV3(x2.contiguous().view(num_batches * length, -1))
        #x3 = x3.contiguous().view(num_batches * length, -1)
        #x4 = x4.contiguous().view(num_batches * length, -1)

        x_K = torch.stack((x0, x1, x2), dim=-2)
        x_Q = torch.stack((xq0, xq1, xq2), dim=-2)
        x_V = torch.stack((xv0, xv1, xv2), dim=-2)

        x_QT = x_Q.permute(0, 2, 1)

        scores = torch.matmul(x_K, x_QT) / math.sqrt(64)

        scores = nn.functional.softmax(scores, dim=-1)

        out = torch.matmul(scores, x_V)

        out = self.ln(out + x_V)

        out = out.view(out.size()[0], -1)

        x = torch.cat((x, out), dim=-1)
        # print("x.shape", x.shape)
        x = self.regressor(x)
        # print("x.shape", x.shape)
        x = x.view(num_batches, length, -1)
        # print("x.shape", x.shape)
        return x


class my_2d1ddy(nn.Module):
    def __init__(self, backbone_state_dict, backbone_mode="ir", modality=['frame'], embedding_dim=512, channels=None, attention=0,
                 output_dim=1, kernel_size=5, dropout=0.1, root_dir='', input_dim_other=[128, 39]):
        super().__init__()

        self.modality = modality
        self.backbone_state_dict = backbone_state_dict
        self.backbone_mode = backbone_mode
        self.root_dir = root_dir

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.attention = attention
        self.dropout = dropout
        self.other_feature_dim = input_dim_other  # input dimension of other modalities

    def init(self, fold=None):
        path = os.path.join(self.root_dir, self.backbone_state_dict + ".pth")

        spatial = my_res50(mode=self.backbone_mode, root_dir=self.root_dir, use_pretrained=False)
        state_dict = torch.load(path, map_location='cpu')
        spatial.load_state_dict(state_dict)

        for param in spatial.parameters():
            param.requires_grad = False

        self.spatial = spatial.backbone
        self.temporal = TemporalConvNet(
            num_inputs=self.embedding_dim, num_channels=self.channels, kernel_size=self.kernel_size, attention=self.attention,
            dropout=self.dropout)

        self.temporal1 = TemporalConvNet(
            num_inputs=self.other_feature_dim[0], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal2 = TemporalConvNet(
            num_inputs=self.other_feature_dim[1], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        '''
        self.temporal3 = TemporalConvNet(
            num_inputs=self.other_feature_dim[2], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        self.temporal4 = TemporalConvNet(
            num_inputs=self.other_feature_dim[3], num_channels=[32, 32, 32, 32], kernel_size=self.kernel_size,
            attention=self.attention,
            dropout=self.dropout)
        '''

        self.encoder1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoder2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoder3 = nn.Linear(32, 32)

        self.encoderQ1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoderQ2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoderQ3 = nn.Linear(32, 32)

        self.encoderV1 = nn.Linear(self.embedding_dim // 4, 32)
        self.encoderV2 = nn.Linear(self.other_feature_dim[0] // 4, 32)
        self.encoderV3 = nn.Linear(32, 32)
        #self.encoder4 = nn.Linear(self.other_feature_dim[3] // 4, 32)
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 32)

        self.ln = nn.LayerNorm([3, 32])

        self.regressor = nn.Linear(224, 8)
        # self.regressor = Sequential(
        #     BatchNorm1d(self.embedding_dim // 4),
        #     Dropout(0.4),
        #     Linear(self.embedding_dim // 4, self.output_dim))

    def forward(self, x, x1, x2):
        print("32323234234")
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)

        if len(x1) > 1 and len(x2) > 1:
            x1 = x1.squeeze().transpose(1, 2).contiguous().float()
            x2 = x2.squeeze().transpose(1, 2).contiguous().float()
        else:
            x1 = x1.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
            x2 = x2.squeeze()[None, :, :].transpose(1, 2).contiguous().float()
        #x3 = x3.transpose(1, 2).contiguous().float()
        #x4 = x4.transpose(1, 2).contiguous().float()

        x1 = self.temporal1(x1).transpose(1, 2).contiguous()
        x2 = self.temporal2(x2).transpose(1, 2).contiguous()
        #x3 = self.temporal3(x3).transpose(1, 2).contiguous()
        #x4 = self.temporal4(x4).transpose(1, 2).contiguous()

        x0 = self.encoder1(x)
        x1 = self.encoder2(x1.contiguous().view(num_batches * length, -1))
        x2 = self.encoder3(x2.contiguous().view(num_batches * length, -1))

        xq0 = self.encoderQ1(x)
        xq1 = self.encoderQ2(x1.contiguous().view(num_batches * length, -1))
        xq2 = self.encoderQ3(x2.contiguous().view(num_batches * length, -1))

        xv0 = self.encoderV1(x)
        xv1 = self.encoderV2(x1.contiguous().view(num_batches * length, -1))
        xv2 = self.encoderV3(x2.contiguous().view(num_batches * length, -1))
        #x3 = x3.contiguous().view(num_batches * length, -1)
        #x4 = x4.contiguous().view(num_batches * length, -1)

        x_K = torch.stack((x0, x1, x2), dim=-2)
        x_Q = torch.stack((xq0, xq1, xq2), dim=-2)
        x_V = torch.stack((xv0, xv1, xv2), dim=-2)

        x_QT = x_Q.permute(0, 2, 1)

        scores = torch.matmul(x_K, x_QT) / math.sqrt(32)

        scores = nn.functional.softmax(scores, dim=-1)

        out = torch.matmul(scores, x_V)

        out = self.ln(out + x_V)

        out = out.view(out.size()[0], -1)

        x = torch.cat((x, out), dim=-1)
        # print("x.shape", x.shape)
        x = self.regressor(x)
        # print("x.shape", x.shape)
        x = x.view(num_batches, length, -1)
        # print("x.shape", x.shape)
        return x
