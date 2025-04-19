import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.model = nn.ModuleList()

        previous_size = configs.d_model
        for hidden_size in configs.hidden_sizes:
            self.model.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size

        self.pred_len = configs.pred_len
        self.hidden_sizes = configs.hidden_sizes
        self.projection = nn.Linear(previous_size, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [B, L, D]
        for i in range(len(self.hidden_sizes)):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc)
#         print('Linear enc_out.shape: ', enc_out.shape)
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'short_term_forecast_earnings' or self.task_name == 't_short_term_forecast_earnings':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len]
        else:
            raise ValueError('Only forecast tasks implemented yet')
