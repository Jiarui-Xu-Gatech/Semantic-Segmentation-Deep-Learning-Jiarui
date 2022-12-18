from typing import Tuple

import torch
from torch import nn

import src.vision.cv2_transforms as transform
from src.vision.part5_pspnet import PSPNet
from src.vision.part4_segmentation_net import SimpleSegmentationNet


def get_model_and_optimizer(args) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create your model, optimizer and configure the initial learning rates.

    Use the SGD optimizer, use a parameters list, and set the momentum and
    weight decay for each parameter group according to the parameter values
    in `args`.

    Create 5 param groups for the 0th + 1st,2nd,3rd,4th ResNet layer modules,
    and then add separate groups afterwards for the classifier and/or PPM
    heads.

    You should set the learning rate for the resnet layers to the base learning
    rate (args.base_lr), and you should set the learning rate for the new
    PSPNet PPM and classifiers to be 10 times the base learning rate.

    Args:
        args: object containing specified hyperparameters, including the "arch"
           parameter that determines whether we should return PSPNet or the
           SimpleSegmentationNet
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    if args.arch == "PSPNet":
        model = PSPNet(layers = args.layers,bins=(1, 2, 3, 6),dropout = 0.1,pretrained=args.pretrained,num_classes=args.classes,
                       criterion=nn.CrossEntropyLoss(ignore_index=255),zoom_factor = args.zoom_factor,use_ppm=args.use_ppm,deep_base = True)
        parameter_list = [{"params": model.layer0.parameters(),"lr": args.base_lr},
                          {"params": model.layer1.parameters(),"lr": args.base_lr},
                          {"params": model.layer2.parameters(),"lr": args.base_lr},
                          {"params": model.layer3.parameters(),"lr": args.base_lr},
                          {"params": model.layer4.parameters(),"lr": args.base_lr},
                          {"params": model.cls.parameters(),"lr": 10 * args.base_lr},
                          {'params': model.ppm.parameters(), 'lr': 10 * args.base_lr},
                          {'params': model.aux.parameters(), 'lr': 10 * args.base_lr}]

        optimizer = torch.optim.SGD(parameter_list,lr = args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay)

    if args.arch == "SimpleSegmentationNet":
        model = SimpleSegmentationNet(pretrained=args.pretrained,criterion=nn.CrossEntropyLoss(ignore_index=args.ignore_label),num_classes=args.classes,deep_base = True)
        parameter_list = [{"params": model.layer0.parameters(),"lr": args.base_lr},
                          {"params": model.resnet.layer1.parameters(),"lr": args.base_lr},
                          {"params": model.resnet.layer2.parameters(),"lr": args.base_lr},
                          {"params": model.resnet.layer3.parameters(),"lr": args.base_lr},
                          {"params": model.resnet.layer4.parameters(),"lr": args.base_lr},
                          {"params": model.cls.parameters(),"lr": 10 * args.base_lr}]
        optimizer = torch.optim.SGD(parameter_list,lr = args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay)


    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer


def update_learning_rate(current_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Given an updated current learning rate, set the ResNet modules to this
    current learning rate, and the classifiers/PPM module to 10x the current
    lr.

    Hint: You can loop over the dictionaries in the optimizer.param_groups
    list, and set a new "lr" entry for each one. They will be in the same order
    you added them above, so if the first N modules should have low learning
    rate, and the next M modules should have a higher learning rate, this
    should be easy modify in two loops.

    Note: this depends upon how you implemented the param groups above.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    i = 0
    for p in optimizer.param_groups:
        if i < 5:
            p['lr'] = current_lr
        else:
            p['lr'] = current_lr * 10


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return optimizer


def get_train_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the training split, with data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    then random horizontal flipping, blurring, rotation, scaling (in any order),
    followed by taking a random crop of size (args.train_h, args.train_w), converting
    the Numpy array to a Pytorch tensor, and then normalizing by the
    Imagenet mean and std (provided here).

    Note that your scaling should be confined to the [scale_min,scale_max] params in the
    args. Also, your rotation should be confined to the [rotate_min,rotate_max] params.

    To prevent black artifacts after a rotation or a random crop, specify the paddings
    to be equal to the Imagenet mean to pad any black regions.

    You should set such artifact regions of the ground truth to be ignored.

    Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    Args:
        args: object containing specified hyperparameters

    Returns:
        train_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    train_transform = transform.Compose([
        transform.ResizeShort(args.short_size),
        transform.RandomHorizontalFlip(),
        transform.RandomGaussianBlur(),
        transform.RandRotate([args.rotate_min, args.rotate_max],padding=mean,ignore_label=args.ignore_label),
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.Crop([args.train_h, args.train_w],crop_type='rand',padding=mean,ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return train_transform


def get_val_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the val split, with no data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    taking a *center* crop of size (args.train_h, args.train_w) with a padding equal
    to the Imagenet mean, converting the Numpy array to a Pytorch tensor, and then
    normalizing by the Imagenet mean and std (provided here).

    Args:
        args: object containing specified hyperparameters

    Returns:
        val_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    val_transform = transform.Compose([
        transform.ResizeShort(args.short_size),
        transform.Crop(size = min(args.train_h, args.train_w),padding=mean,ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return val_transform
