from data_provider.data_loader import Dataset_Text_Earnings, Dataset_TimeSeires_Earnings
from torch.utils.data import DataLoader

data_dict = {
    'text_earnings': Dataset_Text_Earnings,
    'ts_earnings': Dataset_TimeSeires_Earnings,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    if args.task_name == 'text_earnings':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            emb_vendor=args.emb_vendor,
            emb_file=args.emb_file,
            y_column=args.y_column,
            rolling_test=args.rolling_test,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    if args.task_name == 'ts_earnings':
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            ts_pattern=args.ts_pattern,
            scale=args.scale,
            rolling_test=args.rolling_test,
            enc_in=args.enc_in,
            prediction_window=args.prediction_window,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

