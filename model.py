import torch
import numpy as np, torch.nn as nn, torch.nn.functional as F
from utils import cal_accuracy

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, **kwargs):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model=d_model
        self.head_first=12

        # self.w_qs = nn.Linear(d_model, n_head * d_k)
        # self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_qs = nn.Linear(d_model, self.head_first * d_model)
        self.w_ks = nn.Linear(d_model, self.head_first * d_model)
        self.w_qs2 = nn.Linear(d_model, n_head * d_k)
        self.w_ks2 = nn.Linear(d_model, n_head * d_k)
        self.w_qs3 = nn.Linear(d_model, n_head * d_k)
        self.w_ks3 = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   dropout=kwargs['dr_sdp'], head = n_head,d_model=d_model,d_k=d_k)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_in = nn.LayerNorm((3, n_head * d_v))

        # self.fc = nn.Linear(n_head * d_v, d_model)
        self.fc = nn.Linear(self.head_first * d_model, d_model)
        self.fc2 = nn.Linear(self.head_first * d_model, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(kwargs['dr_mha'])
        
        self.src_word_emb = nn.Embedding(201, d_k)
        self.position_enc = PositionalEncoding(d_k, n_position=201)
        
        self.attention2 = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   dropout=kwargs['dr_sdp'], head = n_head,d_model=d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.out_channels=64
        self.kernel_size=1
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1,self.out_channels, (self.kernel_size, 2))  #self.config.out_channels:64 self.config.kernel_size:1 kernel size x 3
        self.conv2_bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(0.3)
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear((self.d_model - self.kernel_size + 1) * self.out_channels, self.d_model, bias=False)

        self.src_word_emb2 = nn.Embedding(201, d_k)
        self.position_enc2 = PositionalEncoding(d_k, n_position=201)
        
        self.w_1 = nn.Conv2d(self.d_model, self.d_model, (1,2))
        self.w_2 = nn.Conv2d(self.d_model, self.d_model, (1,1))
        self.a_h = nn.Parameter(torch.tensor(0.0))
        self.b_h = nn.Parameter(torch.tensor(1.0))
        
        self.a_r = nn.Parameter(torch.tensor(0.0))
        self.b_r = nn.Parameter(torch.tensor(1.0))
        
        self.a_t = nn.Parameter(torch.tensor(1.0))
        self.b_t = nn.Parameter(torch.tensor(0.0))



    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        # residual = v
        residual2 = k
        
      
        q = self.w_qs(q).view(sz_b, len_q, self.head_first, self.d_model)
        k = self.w_ks(k).view(sz_b, len_k, self.head_first, self.d_model)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # v = self.w_vs(q).view(sz_b, len_q, n_head, d_v)

        q = q.transpose(1, 2).contiguous().view(-1, len_q, self.d_model)  # (n*b) x lq x dm
        k = k.transpose(1, 2).contiguous().view(-1, len_k, self.d_model)  # (n*b) x lk x dm
        # v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        re_q=q
        re_k=k
        q = self.w_qs2(q).view(sz_b*self.head_first, len_q, n_head, d_k)
        k = self.w_ks2(k).view(sz_b*self.head_first, len_k, n_head, d_k)
        
        
         #位置嵌入
        a = self.n_head
        indices = torch.arange(1, a+1).unsqueeze(0)
        # indices = indices.to(torch.float32)
        indices=indices.to('cuda')
        # 将整数序列复制batch次
        # print(indices.shape)
        indices = indices.repeat(sz_b*self.head_first, 1)
        # print(indices.shape)
        indices_re=indices
        indices=self.src_word_emb(indices)
        indices=self.position_enc(indices)
        # print(indices.shape)
        q = q.transpose(1, 2).contiguous().squeeze(2)+indices#b h dim
        k = k.transpose(1, 2).contiguous().squeeze(2)+indices
        # v = v.transpose(1, 2).contiguous().squeeze(2)+indices

        # v = v.transpose(1, 2).contiguous().squeeze(2)

        output = self.attention(q, k, v).unsqueeze(1)
        # output =output + self.a_t*re_q
        output =output + re_q
        # output = self.layer_norm(output)#new_r

        
        
        q2 = self.w_qs3(re_q).view(sz_b*self.head_first, len_q, n_head, d_k)
        k2 = self.w_ks3(re_k).view(sz_b*self.head_first, len_k, n_head, d_k)
        
        
        indices2=self.src_word_emb2(indices_re)
        indices2=self.position_enc2(indices2)
        # print(indices.shape)
        q2 = q2.transpose(1, 2).contiguous().squeeze(2)+indices2#b h dim
        k2 = k2.transpose(1, 2).contiguous().squeeze(2)+indices2
        # v = v.transpose(1, 2).contiguous().squeeze(2)+indices

        # v = v.transpose(1, 2).contiguous().squeeze(2)

        output2 = self.attention2(k, q, v).unsqueeze(1)
        # output2 = output2 + self.b_t*re_k
        # output = self.attention(q, k, v)

        output = output.view(sz_b, self.head_first, self.d_model)
        output = output.transpose(1, 2).contiguous().view(sz_b, -1)  # b x lq x (n*dv)
      
        output = self.dropout(self.fc(output)).unsqueeze(1)
        
        output2 = output2.view(sz_b, self.head_first, self.d_model)
        output2 = output2.transpose(1, 2).contiguous().view(sz_b, -1)  # b x lq x (n*dv)
       
        output2 = self.dropout(self.fc2(output2)).unsqueeze(1)
        
        # output = self.layer_norm(-output + residual)#new_h
        output = self.layer_norm(-self.a_h*output + residual)#new_r
        output2 = self.layer_norm(-self.a_r*output2 + residual2)#new_h
      

        output=self.b_r*output2
      
        

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.2,head=64,d_model=256,d_k=32):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=2)
        self.softmax = nn.Softmax(dim=3)
        self.head=head
        self.d_model=d_model
        self.d_k=d_k
        # self.tline=nn.Linear(self.head*self.head,self.d_model)
        self.tline=nn.Linear(self.head*self.head,self.d_model)
        self.bn1 = torch.nn.BatchNorm1d(self.d_model)
        self.trans=nn.Linear(self.head*self.d_k,self.d_model)
        self.trans2=nn.Linear(self.head,self.d_model)
        self.trans3=nn.Linear(self.head*self.d_k,self.d_model)
        self.tline2=nn.Linear(self.head*self.head,self.d_model)
        self.bn2 = torch.nn.BatchNorm1d(self.d_model)

        
        self.w_qs = nn.Linear(d_model, head * d_k)
        self.w_ks = nn.Linear(d_model, head * d_k)
        self.w_vs = nn.Linear(d_model, head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

    def forward(self, q, k, v):
        new_t = torch.matmul(q, k.transpose(1, 2))# batch head head
        # new_t = torch.bmm(q, k.transpose(1, 2))# batch head head
        new_t=new_t.reshape(-1,self.head*self.head)
        new_t=self.tline(new_t)
        new_t = self.bn1(new_t)
        
      
        return new_t


class PositionwiseFeedForwardUseConv(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3, decode_method='twomult'):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        # self.w_1 = nn.Linear(d_in, d_hid)
        # self.w_1 = nn.Conv1d(d_in, d_in, 1)
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        # self.w_2 = nn.Linear(d_hid, d_in)
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.decode_method = decode_method

    def forward(self, x):
        residual = x# b 2 model
        # output=x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        # output = self.w_2(self.w_1(output))
        # output = F.relu(output)
        output = output.transpose(1, 2)
        # output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        if self.decode_method == 'tucker':
            output = output + residual
        else:
            # output = self.layer_norm(output + residual)
            output = output + residual
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, **kwargs):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, **kwargs)
        self.pos_ffn = PositionwiseFeedForwardUseConv(
            d_model, d_inner, dropout=kwargs['dr_pff'], 
            decode_method=kwargs['decoder'])

    def forward(self, enc_input): 
        h,r=torch.chunk(enc_input, 2, dim = 1)# b 1 dim
        # h=h.squeeze(1)# b dim
        # r=r.squeeze(1)
        enc_output = self.slf_attn(h, r, enc_input)

        
        # enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output


class Encoder(nn.Module):
    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, **kwargs):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(kwargs['dr_enc'])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, **kwargs)
            for _ in range(n_layers)])

    def forward(self, edges):
        enc_output = self.dropout(edges)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class TuckER(nn.Module):
    def __init__(self, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-0.05, 0.05, (d2, d1, d1)), dtype=torch.float, requires_grad=True)
        )
        self.bn0 = torch.nn.BatchNorm1d(d1)

    def forward(self, e1, r):
        x = self.bn0(e1)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))

        return x


class SAttLE(nn.Module):

    def __init__(
            self, num_nodes,
            num_rels, n_layers,
            d_embd, d_k, d_v,
            d_model, d_inner, n_head,
            label_smoothing=0.1,
            **kwargs
    ):
        super(SAttLE, self).__init__()

        self.encoder = Encoder(d_embd, n_layers, n_head, d_k, d_v, d_model, d_inner, **kwargs)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_nodes)))
        self.num_nodes, self.num_rels = num_nodes, num_rels
        self.d_embd = d_embd
        self.init_parameters()
        if kwargs['decoder'] == 'tucker':
            print('decoding with tucker mode!')
            self.decode_method = 'tucker'
            self.tucker = TuckER(d_embd, d_embd)
        else:
            self.decode_method = 'twomult'
        self.ent_bn = nn.BatchNorm1d(d_embd)
        self.rel_bn = nn.BatchNorm1d(d_embd)
        self.label_smoothing = label_smoothing

    def init_parameters(self):
        self.tr_ent_embedding = nn.Parameter(torch.Tensor(self.num_nodes, self.d_embd))
        self.rel_embedding = nn.Parameter(torch.Tensor(2 * self.num_rels, self.d_embd))
        nn.init.xavier_normal_(self.tr_ent_embedding.data)
        nn.init.xavier_normal_(self.rel_embedding.data)

    def cal_loss(self, scores, edges, label_smooth=True):
        labels = torch.zeros_like((scores), device='cuda')
        labels.scatter_(1, edges[:, 2][:, None], 1.)
        if self.label_smoothing:
            labels = ((1.0 - self.label_smoothing) * labels) + (1.0 / labels.size(1))
        pred_loss = F.binary_cross_entropy_with_logits(scores, labels)

        return pred_loss

    def cal_score(self, edges, mode='train'):
        h = self.tr_ent_embedding[edges[:, 0]]
        r = self.rel_embedding[edges[:, 1]]
        h = self.ent_bn(h)[:, None]
        r = self.rel_bn(r)[:, None]#b 1 dim
        # feat_edges = torch.hstack((h, r))
        feat_edges = torch.cat([h, r],dim=1)


        embd_edges = self.encoder(feat_edges)
        if self.decode_method == 'tucker':
            src_edges = self.tucker(embd_edges[:, 0, :], embd_edges[:, 1, :])
        else:
            # src_edges = embd_edges[:, 1, :]
            src_edges = embd_edges.squeeze(1)
        scores = torch.mm(src_edges, self.tr_ent_embedding.transpose(0, 1))
        # scores = torch.matmul(src_edges, self.tr_ent_embedding.transpose(0, 1))
        scores += self.b.expand_as(scores)
        return scores

    def forward(self, feat_edges):
        scores = self.cal_score(feat_edges, mode='train')
        return scores

    def predict(self, edges):
        labels = edges[:, 2]
        edges = edges[0][None, :]
        scores = self.cal_score(edges, mode='test')
        scores = scores[:, labels.view(-1)].view(-1)
        scores = torch.sigmoid(scores)
        acc = 0.0
        return scores, acc


