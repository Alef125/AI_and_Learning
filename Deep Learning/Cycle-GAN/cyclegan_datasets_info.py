"""Contains the cyclegan data."""

horse2zebra_Train_INFO = {
    'SIZE': 1334,
    'IMAGE_TYPE': '.jpg',
    'PATH_TO_GROUP_A': './inputs/horse2zebra/trainA',
    'PATH_TO_GROUP_B': './inputs/horse2zebra/trainB',
    'PATH_TO_CSV': './inputs/horse2zebra/horse2zebra_train.csv',
    'LOG_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/horse2zebra',
    'CHECKPOINT_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/horse2zebra/exp_01'
}

horse2zebra_Test_INFO = {
    'SIZE': 140,
    'IMAGE_TYPE': '.jpg',
    'PATH_TO_GROUP_A': './inputs/horse2zebra/testA',
    'PATH_TO_GROUP_B': './inputs/horse2zebra/testB',
    'PATH_TO_CSV': './inputs/horse2zebra/horse2zebra_test.csv',
    'LOG_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/horse2zebra',
    'CHECKPOINT_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/horse2zebra/exp_01'
}

FERG_Train_INFO = {
    'SIZE': 6000,
    'IMAGE_TYPE': '.png',
    'PATH_TO_GROUP_A': './inputs/FERG/trainA',
    'PATH_TO_GROUP_B': './inputs/FERG/trainB',
    'PATH_TO_CSV': './inputs/FERG/FERG_train.csv',
    'LOG_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/FERG',
    'CHECKPOINT_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/FERG/exp_01'
}

FERG_Test_INFO = {
    'SIZE': 1100,
    'IMAGE_TYPE': '.png',
    'PATH_TO_GROUP_A': './inputs/FERG/testA',
    'PATH_TO_GROUP_B': './inputs/FERG/testB',
    'PATH_TO_CSV': './inputs/FERG/FERG_test.csv',
    'LOG_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/FERG',
    'CHECKPOINT_DIR': '/Users/macintosh/Desktop/facialGAN/outputs/FERG/exp_01'
}


def get_dataset_info(dataset_name, info):
    valid_info = ['SIZE', 'IMAGE_TYPE', 'PATH_TO_CSV', 'PATH_TO_GROUP_A', 'PATH_TO_GROUP_B',
                  'LOG_DIR', 'CHECKPOINT_DIR']
    if info not in valid_info:
        raise ValueError('Requested info %s was not valid.' % info)

    if dataset_name == 'horse2zebra_train':
        return horse2zebra_Train_INFO[info]
    elif dataset_name == 'horse2zebra_train':
        return horse2zebra_Test_INFO[info]
    elif dataset_name == 'FERG_train':
        return FERG_Train_INFO[info]
    elif dataset_name == 'FERG_test':
        return FERG_Test_INFO[info]
    else:
        raise ValueError('dataset name %s was not recognized.' % dataset_name)
