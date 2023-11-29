import argparse
from Diffusion.Train import train, eval
import os
def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training and Evaluation")
    parser.add_argument("--state", type=str, choices=["train", "eval"], default="train", help="Training or evaluation mode")
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
    parser.add_argument("--T", type=int, default=1000, help="Diffusion steps")
    parser.add_argument("--channel", type=int, default=128, help="Channel size")
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 3, 4], help="Channel multiplier")
    parser.add_argument("--attn", type=int, nargs="+", default=[2], help="Attention layers")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--multiplier", type=float, default=2.0, help="Multiplier")
    parser.add_argument("--beta_1", type=float, default=1e-4, help="Beta 1")
    parser.add_argument("--beta_T", type=float, default=0.02, help="Beta T")
    parser.add_argument("--img_size", type=int, default=32, help="Image size")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g., 'cuda:0')")
    parser.add_argument("--training_load_weight", type=str, default=None, help="Path to training weights")
    parser.add_argument("--save_weight_dir", type=str, default="./Checkpoints/", help="Directory to save weights")
    parser.add_argument("--test_load_weight", type=str, default="ckpt_199_.pt", help="Path to test weights")
    parser.add_argument("--sampled_dir", type=str, default="./SampledImgs/", help="Directory to save sampled images")
    parser.add_argument("--sampledNoisyImgName", type=str, default="NoisyNoGuidenceImgs.png", help="Noisy image name")
    parser.add_argument("--sampledImgName", type=str, default="SampledNoGuidenceImgs.png", help="Sampled image name")
    parser.add_argument("--nrow", type=int, default=8, help="Number of rows for sampled images")
    parser.add_argument("--save_middle_result", type=bool, default=False, help="Save middle result")
    return parser.parse_args()

def main():
    args = parse_args()
    # if the sampled_dir does not exist, create it
    if not os.path.exists(args.sampled_dir):
        os.makedirs(args.sampled_dir)
    if args.state == "train":
        train(vars(args))
    else:
        eval(vars(args))

if __name__ == '__main__':
    main()
