from torchvision import transforms
from .data_utils import CUB, dogs
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

PretrainMeans = [0.485, 0.456, 0.406]
PretrainStds = [0.229, 0.224, 0.225]
CalMeans = [0.4856, 0.4994, 0.4324]
CalStds = [0.1817, 0.1811, 0.1927]
imagesize = 224

def get_loader(args):
    if args.dataset == 'CUB_200_2011':
        if args.pretrained:
            mean = PretrainMeans
            std = PretrainStds
        if not args.pretrained:
            if args.model_name == "transfg":
                mean = PretrainMeans
                std = PretrainStds
            else:
                mean = CalMeans
                std = CalStds

        train_transform = transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(args.img_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = mean, 
                                                std = std)
                        ])
        test_transform = transforms.Compose([
                                    transforms.Resize(args.img_size),
                                    transforms.CenterCrop(args.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = mean, 
                                                        std = std)
                        ])
        train_data = CUB(root=args.data_root, is_train=True, transform = train_transform)
        test_data = CUB(root=args.data_root, is_train=False, transform = test_transform)
        
    if args.dataset == 'dog':
        mean = PretrainMeans
        std = PretrainStds
        train_transform = transforms.Compose([
                            transforms.Resize(args.img_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(args.img_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = mean, 
                                                std = std)
                        ])
        test_transform = transforms.Compose([
                                    transforms.Resize(args.img_size),
                                    transforms.CenterCrop(args.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = mean, 
                                                        std = std)
                        ])
        train_data = dogs(root=args.data_root, train=True, transform = train_transform)
        test_data = dogs(root=args.data_root, train=False, transform = test_transform)

    train_loader = data.DataLoader(train_data, 
                                   shuffle = True, 
                                   batch_size = args.train_batch_size)
    test_loader = data.DataLoader(test_data, 
                                  batch_size = args.eval_batch_size)
    return train_loader, test_loader

# def get_loader(args):

#     if args.dataset == 'CUB_200_2011':
#         train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#                                     transforms.RandomCrop((448, 448)),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#                                     transforms.CenterCrop((448, 448)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
#         testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
#     # elif args.dataset == 'car':
#     #     trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
#     #                         os.path.join(args.data_root,'cars_train'),
#     #                         os.path.join(args.data_root,'devkit/cars_meta.mat'),
#     #                         # cleaned=os.path.join(data_dir,'cleaned.dat'),
#     #                         transform=transforms.Compose([
#     #                                 transforms.Resize((600, 600), Image.BILINEAR),
#     #                                 transforms.RandomCrop((448, 448)),
#     #                                 transforms.RandomHorizontalFlip(),
#     #                                 AutoAugImageNetPolicy(),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #                         )
#     #     testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
#     #                         os.path.join(args.data_root,'cars_test'),
#     #                         os.path.join(args.data_root,'devkit/cars_meta.mat'),
#     #                         # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
#     #                         transform=transforms.Compose([
#     #                                 transforms.Resize((600, 600), Image.BILINEAR),
#     #                                 transforms.CenterCrop((448, 448)),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #                        )
#     elif args.dataset == 'dog':
#         train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#                                     transforms.RandomCrop((448, 448)),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#                                     transforms.CenterCrop((448, 448)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         trainset = dogs(root=args.data_root,
#                                 train=True,
#                                 cropped=False,
#                                 transform=train_transform,
#                                 download=False
#                                 )
#         testset = dogs(root=args.data_root,
#                                 train=False,
#                                 cropped=False,
#                                 transform=test_transform,
#                                 download=False
#                                 )
#     # elif args.dataset == 'nabirds':
#     #     train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#     #                                     transforms.RandomCrop((448, 448)),
#     #                                     transforms.RandomHorizontalFlip(),
#     #                                     transforms.ToTensor(),
#     #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #     test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
#     #                                     transforms.CenterCrop((448, 448)),
#     #                                     transforms.ToTensor(),
#     #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #     trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
#     #     testset = NABirds(root=args.data_root, train=False, transform=test_transform)
#     # elif args.dataset == 'INat2017':
#     #     train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
#     #                                 transforms.RandomCrop((304, 304)),
#     #                                 transforms.RandomHorizontalFlip(),
#     #                                 AutoAugImageNetPolicy(),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #     test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
#     #                                 transforms.CenterCrop((304, 304)),
#     #                                 transforms.ToTensor(),
#     #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     #     trainset = INat2017(args.data_root, 'train', train_transform)
#     #     testset = INat2017(args.data_root, 'val', test_transform)

    

#     train_sampler = RandomSampler(trainset)
#     test_sampler = SequentialSampler(testset)
#     train_loader = DataLoader(trainset,
#                               sampler=train_sampler,
#                               batch_size=args.train_batch_size)
#     test_loader = DataLoader(testset,
#                              sampler=test_sampler,
#                              batch_size=args.eval_batch_size) if testset is not None else None

#     return train_loader, test_loader