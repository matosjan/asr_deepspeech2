import argparse
import pathlib
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

def calc_metrics(pred_dir, gt_dir):
    wer = 0
    cer = 0
    file_counter = 0
    gt_dir = Path(gt_dir)
    for path in Path(pred_dir).iterdir():
        filename = pathlib.PurePath(path)

        target_text = None
        with open(gt_dir / filename, 'r') as f_gt:
            target_text = f_gt.read()
        target_text = CTCTextEncoder.normalize_text(target_text)

        predicted_text = None
        with open(path, 'r') as f_pred:
            predicted_text = f_pred.read()
        
        cer += calc_cer(target_text, predicted_text)
        wer += calc_wer(target_text, predicted_text)
        file_counter += 1
    
    print(f"CER: {(cer / file_counter):.4f}")
    print(f"WER: {(wer / file_counter):.4f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pred_dir", required=True, type=str, help="Path to directory with predictions")
    args.add_argument("--gt_dir", required=True, type=str, help="Path to directory with ground truths")
    args = args.parse_args()
    calc_metrics(pred_dir=args.pred_dir, gt_dir=args.gt_dir)

