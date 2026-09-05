import os.path as osp
import argparse
from omegaconf import OmegaConf
from src.engine.trainer_utils import instantiate_from_config
from src.utils.device_utils import configure_compute_backend


def train():
    configure_compute_backend()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vit_vqgan.yaml')
    args = parser.parse_args()
    cfg_file = args.cfg
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    trainer = instantiate_from_config(config.trainer)
    trainer.train(args.cfg)

if __name__ == '__main__':
    train()
