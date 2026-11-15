"""
Nirnay: Fast, Calibrated System One Decision Model powered by Qwen2.5-0.5B.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Nirnay - System One Decision Model",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser("train", help="Run production distillation training")
    train_parser.add_argument("--num_samples", type=int, default=-1, help="Total training samples (-1 for full corpus)")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")

    # pilot command
    pilot_parser = subparsers.add_parser("pilot", help="Run pilot distillation training")
    pilot_parser.add_argument("--samples", type=int, default=10000, help="Number of pilot rows")
    pilot_parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b-pilot")
    pilot_parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Benchmark model against test_set_30k and ood")
    eval_parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b-pilot")
    eval_parser.add_argument("--num_samples", type=int, default=2000, help="Number of evaluation samples")

    # decide command
    decide_parser = subparsers.add_parser("decide", help="Run interactive decision inference CLI")
    decide_parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b-pilot")

    # export command
    export_parser = subparsers.add_parser("export", help="Merge LoRA adapters into standalone safetensors")
    export_parser.add_argument("--adapter_dir", type=str, default="checkpoints/nirnay-0.5b-pilot")
    export_parser.add_argument("--output_dir", type=str, default="checkpoints/nirnay-0.5b-merged")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run /v1/systemone HTTP server for S1MB harness")
    serve_parser.add_argument("--model_path", type=str, default="checkpoints/nirnay-0.5b/checkpoint-latest")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    if args.command == "train":
        from scripts.train_full import run_full_distillation
        run_full_distillation(
            num_train_samples=args.num_samples,
            epochs=args.epochs,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
        )
    elif args.command == "pilot":
        from scripts.train_pilot import run_pilot_distillation
        run_pilot_distillation(
            num_train_samples=args.samples,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
        )
    elif args.command == "evaluate":
        from scripts.run_eval import run_benchmarks
        run_benchmarks(
            model_path=args.model_path,
            num_eval_samples=args.num_samples,
        )
    elif args.command == "decide":
        from src.infer import NirnayEngine
        engine = NirnayEngine.from_pretrained(args.model_path)
        print("\nNirnay Engine loaded. Ready for single-pass decisions.")
    elif args.command == "export":
        from src.export import export_merged_model
        export_merged_model(
            base_model_name="Qwen/Qwen2.5-0.5B",
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "serve":
        from scripts.serve import serve
        serve(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
        )

if __name__ == "__main__":
    main()
