import unittest
import torch
from torch.utils.data.dataloader import DataLoader

from datasets.dtu_yao import MVSDataset

class TestDataloader(object):

    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        # super().__init__()
        self.mvs_dataset = MVSDataset(datapath, listfile, mode, nviews, ndepths=ndepths,
                                      interval_scale=interval_scale, **kwargs)
        self.args = kwargs

    def test_get_item(self):
        print(len(self.mvs_dataset))
        for i in range(len(self.mvs_dataset)):
            sample = self.mvs_dataset[i]
            print(sample['is_begin'])

    def test_dataloader(self):
        train_sampler = torch.utils.data.SequentialSampler(self.mvs_dataset)
        TrainImgLoader = DataLoader(self.mvs_dataset, batch_size=self.args['batch_size'], shuffle=False,
                                    sampler=train_sampler, num_workers=4, pin_memory=True)
        print(len(TrainImgLoader))
        for batch in TrainImgLoader:
            print(batch['is_begin'])


if __name__ == '__main__':
    dataloader = TestDataloader('/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/train',
                                '../../lists/dtu/train.txt', 'train', 5, seq_size=49, batch_size=3, shuffle=False)
    # dataloader.test_get_item()
    dataloader.test_dataloader()

