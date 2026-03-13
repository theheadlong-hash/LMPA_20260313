import sys
import os

# 获取当前脚本（SelfAttention_Family.py）所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录，也就是包含 utils 的目录
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到 sys.path 中
sys.path.append(parent_dir)

try:
    from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Pretrain
    print("data_factory")
except ImportError as e:
    print(f"导入失败：{e}")


from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'pretrain': Dataset_Pretrain,
}

# 更像是一个管理者，它不具体处理数据，而是根据一些条件来决定使用哪个数据处理员（数据集类，在data_loder ）
def data_provider(args, flag):
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
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader
