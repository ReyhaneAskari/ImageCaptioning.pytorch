import argparse
from dataloader import DataLoader

opt = argparse.Namespace(
    batch_size=32,
    input_att_dir='/data/lisa/data/coco_preprocessed/data/cocotalk_att',
    input_fc_dir='/data/lisa/data/coco_preprocessed/data/cocotalk_fc',
    input_json='/data/lisa/data/coco_preprocessed/data/cocotalk.json',
    input_label_h5='/data/lisa/data/coco_preprocessed/data/cocotalk_label.h5',
    seq_per_img=5, use_att=True, train_only=0)
loader = DataLoader(opt)

# Get one minibatch:
i = 0
for i in range(770):
    data = loader.get_batch('train')
    if i % 10 == 0:
        print i
