import argparse
import torch
import logging
import random
import numpy as np
import time
import os

from models.configs import CONFIGS
from models.modeling import VGG, GoogleNet, ResNet, inception_v3
from utils.dataset import get_loader
from utils.scheduler import EarlyStopping
from models.transfg import VisionTransformer

import torch.nn as nn
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)

tr_loss = []
ts_loss = []
tr_acc1 = []
ts_acc1 = []
tr_acc5 = []
ts_acc5 = []
tr_loss = []

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def setup(args):
    config = CONFIGS[args.model_name]
    if args.model_name == "transfg":
        config.split = args.split
        config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "dog":
        num_classes = 120
    print("pretrain:", args.pretrained)

    if args.pretrained:
        print("Use Pretrained Model!")
        if "resnet" in args.model_name:
            pretrained_model = models.resnet50(pretrained = True)
            IN_FEATURES = pretrained_model.fc.in_features 
            fc = nn.Linear(IN_FEATURES, num_classes)
            pretrained_model.fc = fc
        elif "vgg" in args.model_name:
            pretrained_model = models.vgg19_bn(pretrained = True)
            IN_FEATURES = pretrained_model.classifier[-1].in_features
            final_fc = nn.Linear(IN_FEATURES, num_classes)
            pretrained_model.classifier[-1] = final_fc
        elif "googlenet" in args.model_name:
            pretrained_model = models.googlenet(pretrained = True)
            IN_FEATURES = pretrained_model.fc.in_features
            final_fc = nn.Linear(IN_FEATURES, num_classes)
            pretrained_model.fc = final_fc
        elif "inceptionv3" in args.model_name:
            pretrained_model = models.inception_v3(pretrained = True)
            IN_FEATURES = pretrained_model.fc.in_features
            final_fc = nn.Linear(IN_FEATURES, num_classes)
            pretrained_model.fc = final_fc
        model = pretrained_model

    if not args.pretrained:
        print("Use Original Model!")
        if "vgg" in args.model_name:
            model = VGG(config, num_classes)
        elif "resnet" in args.model_name:
            model = ResNet(config, num_classes)
        elif "googlenet" in args.model_name:
            model = GoogleNet(3, num_classes)
        elif "transfg" in args.model_name:
            model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
            model.load_from(np.load(args.pretrained_dir))
        elif "inceptionv3" in args.model_name:
            model = inception_v3(num_classes = num_classes, transform_input = True)

    print("Load Gpu")
    model.to(args.device)
    #print("Count Parameter")
    #num_params = count_parameters(model)

    logger.info("{} set up successful!".format(args.model_name))
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)   
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def train_epoch(model, iterator, optimizer, criterion, scheduler, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        if (isinstance(y_pred, tuple)):
            y_pred = y_pred[0]
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            if (isinstance(y_pred, tuple)):
                y_pred = y_pred[0]

            loss = criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def train_transfg_epoch(model, iterator, optimizer, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        loss, logits = model(x, y)
        loss = loss.mean()


        # y_pred = model(x)
        # if (isinstance(y_pred, tuple)):
        #     y_pred = y_pred[0]
        
        # loss = criterion(y_pred, y)
        
        acc_1, acc_5 = calculate_topk_accuracy(logits, y)
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def evaluate_transfg(model, iterator, optimizer, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0
    
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            if (isinstance(y_pred, tuple)):
                y_pred = y_pred[0]

            loss = loss_fct(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_5 /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_5

def train(args, model):
    """ Train the model """
    train_loader, test_loader = get_loader(args)
    logger.info("* Load Dataset successful! *")

    if args.model_name == "transfg":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=3e-2,
                                    momentum=0.9,
                                    weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    total_steps = args.epoch * len(train_loader)
    logger.info("Training total steps is {}".format(total_steps))
    # if args.model_name == "transfg":
    #     total_steps = 10000
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=total_steps)
    #scheduler = lr_scheduler.OneCycleLR(optimizer,
                            # max_lr = args.learning_rate,
                            # total_steps = total_steps)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    # best_valid_loss = float('inf')
    best_valid_acc_1 = 0
    # 迭代七次如果不下降则早停
    es = EarlyStopping(patience=10, mode = 'max')

    if args.is_train:
        logger.info("***** Running training *****")
        logger.info("  Total epoch = %d", args.epoch)
        for epoch in range(args.epoch):
            start_time = time.monotonic()
            if args.model_name == "transfg":
                train_loss, train_acc_1, train_acc_5 = train_transfg_epoch(model, train_loader, optimizer, scheduler, args.device)
                valid_loss, valid_acc_1, valid_acc_5 = evaluate_transfg(model, test_loader, optimizer, scheduler, args.device)
            else:
                train_loss, train_acc_1, train_acc_5 = train_epoch(model, train_loader, optimizer, criterion, scheduler, args.device)
                valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, test_loader, criterion, args.device)

            # if valid_loss < best_valid_loss:
            #     best_valid_loss = valid_loss
            #     save_model = '{}/{}'.format(args.output, "model.pt")
            #     torch.save(model.state_dict(), save_model)
            if valid_acc_1 > best_valid_acc_1:
                best_valid_acc_1 = valid_acc_1
                save_model = '{}/{}'.format(args.output, "model.pt")
                torch.save(model.state_dict(), save_model)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
                f'Train Acc @5: {train_acc_5*100:6.2f}%')
            logger.info(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
                f'Valid Acc @5: {valid_acc_5*100:6.2f}%')

            if es.step(torch.tensor(valid_acc_1)):
                logger.info("Early Stop!!!")
                break  # early stop criterion is met, we can stop now

            tr_loss.append(train_loss)
            ts_loss.append(valid_loss)
            tr_acc1.append(train_acc_1)
            ts_acc1.append(valid_acc_1)
            tr_acc5.append(train_acc_5)
            ts_acc5.append(valid_acc_5)
        logger.info("***** Testing *****")
        model.load_state_dict(torch.load(save_model))

    
    test_loss, test_acc_1, test_acc_5 = evaluate(model, test_loader, criterion, args.device)

    logger.info(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
        f'Test Acc @5: {test_acc_5*100:6.2f}%')

def train_save(args):
    tr_loss_np = np.array(tr_loss)
    ts_loss_np = np.array(ts_loss)
    tr_acc1_np = np.array(tr_acc1)
    ts_acc1_np = np.array(ts_acc1)
    tr_acc5_np = np.array(tr_acc5)
    ts_acc5_np = np.array(ts_acc5)
    np.save(args.output + '/tr_loss_np.npy',tr_loss_np)
    np.save(args.output + '/ts_loss_np.npy',ts_loss_np)
    np.save(args.output + '/tr_acc1_np.npy',tr_acc1_np)
    np.save(args.output + '/ts_acc1_np.npy',ts_acc1_np)
    np.save(args.output + '/tr_acc5_np.npy',tr_acc5_np)
    np.save(args.output + '/ts_acc5_np.npy',ts_acc5_np)

def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CUB_200_2011", "dog"], default="CUB_200_2011",
                        help="Which dataset")
    parser.add_argument('--data_root', type=str, default='../datasets')            
    parser.add_argument('--model_name', type=str,
                        choices=["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet151", "googlenet", "inceptionv3", "transfg"],
                        default="resnet50",
                        help="Name of the model")
    parser.add_argument('--pretrained', type=int, default=0,
                        help="Whether to pretrained")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch", default=50, type=int, 
                        help="The number of epochs")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--output', type=str, default="./outputs",
                        help="output directory")
    parser.add_argument('--is_train', default=True,
                        help="output directory")

    parser.add_argument('--split', type=str, default='overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument("--pretrained_dir", type=str, default="./vit_pre/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    args = parser.parse_args()

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    
    if args.pretrained:
        output_path = "Pre_" + args.model_name + "_" + args.dataset
    else:
        output_path = args.model_name + "_" + args.dataset
    output_path = os.path.join(args.output, output_path) 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    args.output = output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logger.info("datasets is in {}".format(args.data_root))
    logger.info("outputpath is in {}".format(args.output))
    logger.info("training device is {}".format(args.device))
    
    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)

    # Save loss and Acc
    train_save(args)

if __name__ == "__main__":
    main()
