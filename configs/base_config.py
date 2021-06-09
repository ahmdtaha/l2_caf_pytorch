import argparse
import os.path as osp
import logging.config
from utils import os_utils
from utils import log_utils

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="L2-CAF pytorch implementation")
        # General Config
        parser.add_argument(
            "--output_dir", help="path to save output heatmaps / log_file", default="./output_heatmaps/"
        )

        parser.add_argument(
            "--log_file", help="What is the name of the log file", default="l2_caf_log.txt"
        )

        parser.add_argument(
            "--input_img", help="Which input image to process for attention visualization", required=True,
        )

        parser.add_argument(
            "--arch", help="Which network architecture to use", default="resnet50",
            choices=['resnet50','googlenet','densenet169'],
        )

        parser.add_argument(
            "--max_iter", help="What is the maximum number of gradient descent iterations",type=int, default=500,
        )

        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.1,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )

        parser.add_argument(
            "--min_error",
            default=10e-7,
            help="When L2-CAF break gradient descent?",
        )

        parser.add_argument(
            "--cls_logits",
            default=None,
            type=lambda x: [int(a) for a in x.split(",")],
            help="Which logits to maximize for the input img?? -- assuming class specific L2-CAF",
        )

        self.parser = parser

    def parse(self,args):
        self.cfg = self.parser.parse_args(args)

        os_utils.touch_dir(self.cfg.output_dir)
        log_file = osp.join(self.cfg.output_dir, self.cfg.log_file)
        logging.config.dictConfig(log_utils.get_logging_dict(log_file))
        self.cfg.logger = logging.getLogger('L2-CAF')

        return self.cfg

