from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Stream_Flow, Dataset_Gaze_Height, Dataset_Gaze_Height_Full, Dataset_Gaze_Height_Mostafa_Socastee
from torch.utils.data import DataLoader
import sys

data_dict = {
    'gaze_height_waycross': Dataset_Gaze_Height_Mostafa_Socastee,
    'gaze_height_socastee': Dataset_Gaze_Height_Mostafa_Socastee,
    'gaze_height_full': Dataset_Gaze_Height_Full,
    'gaze_height': Dataset_Gaze_Height,
    '1Hour': Dataset_Stream_Flow,
    '3Hours': Dataset_Stream_Flow,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    print('woring..............')
    sys.stdout.flush()
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        tempfeat=args.tempfeat
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
