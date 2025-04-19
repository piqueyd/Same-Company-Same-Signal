import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.model = nn.ModuleList()

        self.temporal = nn.Sequential(
            nn.Linear(configs.emb_dim, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.emb_dim),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.emb_dim, configs.pred_len)

    def forecast(self, t_emb, mask=None):
        # x: [B, L, D]
        if len(t_emb.shape) == 2:
            t_emb = t_emb + self.temporal(t_emb)
        else:
            t_emb = t_emb + self.temporal(t_emb.transpose(1, 2)).transpose(1, 2)
            t_emb = t_emb + self.channel(t_emb)

        
        if len(t_emb.shape) == 2:
            t_out = self.projection(t_emb)
        else:
            t_out = self.projection(t_emb.transpose(1, 2)).transpose(1, 2)

        return t_out

    def forward(self, t_emb, mask=None):
        if self.task_name == 'text_earnings':
            t_out = self.forecast(t_emb)
            return t_out[:, -self.pred_len]
        else:
            raise ValueError('Only forecast tasks implemented yet')
