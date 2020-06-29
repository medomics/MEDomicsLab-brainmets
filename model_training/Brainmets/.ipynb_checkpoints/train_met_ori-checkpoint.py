import argparse
from typing import List
from Brainmets.utils import *
from Brainmets.dataset import *
from Brainmets.augmentations import Transformer
from Brainmets.losses import *
from Brainmets.trainer import *
from Brainmets.evaluation import *

if __name__== '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="num of epochs to train", type=int)
    parser.add_argument("--name", help="data mode")
    parser.add_argument("--suffix", help="anything to add on the end of the name of the saved model")
    parser.add_argument("--gpu", help="gpu to train on")
    parser.add_argument("--max_lr", help="max learning rate", type=float)
    parser.add_argument("--loss", help="loss function to use")
    parser.add_argument("--pos_weight", help="pos_weight for WBCE", type=float)
    parser.add_argument("--debug", help="debug mode or not")
    parser.add_argument("--clahe", help="clahe or not", type=bool)
    parser.add_argument("--gamma", help="gamma for focal loss", type=float)
    parser.add_argument("--met_num", help="split the data by number of met", type=list)
    parser.add_argument("--met_size", help="split the data by size of met", type=float)
    parser.add_argument("--met_location", help="split the data by location of met", type=list)
    parser.add_argument("--div_factor", help="div_factor at the beginning", type=int)
    parser.add_argument("--final_div_factor", help="div_factor at the end", type=int)
    args = parser.parse_args()
    
    loss = args.loss
    pos_weight = args.pos_weight
    suffix = args.suffix
    gpu = args.gpu
    debug = args.debug
    debug_size = 20
    init_lr = 0.001
    max_lr = args.max_lr
    final_div_factor = args.final_div_factor
    div_factor = args.div_factor
    
    epochs = args.epochs
    clahe = args.clahe
    print_per_instance = True
    use_one_cycle = True
    gamma = args.gamma
    met_num = args.met_num
    met_size = args.met_size
    met_location = args.met_location
    
    print(met_num)
    
    name = '-'.join([args.name, loss, suffix, str(div_factor), str(final_div_factor)])

    device = torch.device('cuda:' + gpu)

    data_path = Path('~/practicum/MEDomicsLab-develop-brainmets/model_training/3_unet_3d_residual/data')
    data = 'original_met_num_data'
    df = pd.read_csv(data_path/f'{data}.csv')

    train_df = df[df['split'] == 'train'].sample(frac=1)
    valid_df = df[df['split'] == 'valid'].sample(frac=1)
    test_df = df[df['split'] == 'test'].sample(frac=1)
    
    if met_num:
        train_df = train_df[(train_df['met_num']>=int(met_num[0])) & (train_df['met_num']<=int(met_num[1]))]
        valid_df = valid_df[(valid_df['met_num']>=int(met_num[0])) & (valid_df['met_num']<=int(met_num[1]))]
        test_df = test_df[(test_df['met_num']>=int(met_num[0])) & (test_df['met_num']<=int(met_num[1]))]
    
    train_img_files = list(train_df['img_files'])
    train_mask_files = list(train_df['mask_files'])
    valid_img_files = list(valid_df['img_files'])
    valid_mask_files = list(valid_df['mask_files'])
    test_img_files = list(test_df['img_files'])
    test_mask_files = list(test_df['mask_files'])

    img_files = sorted(train_img_files + valid_img_files + test_img_files)
    mask_files = sorted(train_mask_files + valid_mask_files + test_mask_files)
    img_names = ['_'.join(file.split('/')[-1].split('_')[0:2])
                 for file in img_files]
    mask_names = ['_'.join(file.split('/')[-1].split('_')[0:2])
                  for file in mask_files]
    assert img_names == mask_names

    # train_transformer = Transformer(axes=['d', 'h', 'w'], max_zoom_rate=1.5, angle=15)
    train_transformer = None
    valid_transformer = None
    test_transformer = None

    if debug == 'True':
        train_dataset = MetDataSet(
            train_df.iloc[:debug_size], train_transformer, clahe=clahe)
        valid_dataset = MetDataSet(valid_df.iloc[:debug_size], clahe=clahe)
        test_dataset = MetDataSet(test_df.iloc[:debug_size], clahe=clahe)
    else:
        train_dataset = MetDataSet(train_df, train_transformer, clahe=clahe)
        valid_dataset = MetDataSet(valid_df, clahe=clahe)
        test_dataset = MetDataSet(test_df, clahe=clahe)

    print('train data size: ', len(train_dataset))
    print('valid data size: ', len(valid_dataset))
    print('test data size: ', len(test_dataset))

    if loss == 'Diceloss':
        loss_func = DiceLoss().to(device)
    elif loss == 'BCE':
        loss_config = {'name': 'BCEWithLogitsLoss'}
        config = {'loss': loss_config}
        loss_func = get_loss_criterion(config).to(device)
    elif loss == 'BCE_my':
        loss_func = FocalLossLogits().to(device)
    elif loss == 'Focal':
        loss_func = FocalLossLogits(pos_weight=1, gamma=0.5).to(device)
    elif loss == 'WBCE_my':
        loss_func = FocalLossLogits(pos_weight=2).to(device)
    elif loss == 'WBCE':
        pos_weights = torch.tensor(2)
        loss_config = {
            'name': 'WeightedBCEWithLogitsLoss',
            'pos_weight': pos_weights}
        config = {'loss': loss_config}
        loss_func = get_loss_criterion(config).to(device)
    elif loss == 'BCEDiceJaccard':
        from Brainmets.losses2 import *
        weights = {'bce':0.3, 'jacc':0.3, 'dice':0.4}
        loss_func = BCEDiceJaccardLoss(weights).to(device)

    trainer = Trainer(
        name,
        'ResidualUNet3D',
        train_dataset,
        valid_dataset,
        test_dataset,
        1,
        init_lr,
        max_lr,
        loss_func,
        div_factor,
        final_div_factor,
        device)

    trainer.fit(epochs, print_per_instance, use_one_cycle)

#     trainer = Trainer.load_best_checkpoint(name)

#     test_score = trainer.predict()


    plt.plot(trainer.train_losses)
    plt.plot(trainer.valid_scores)
    plt.show()
    plt.savefig('./results/' + name + '/losses.png')