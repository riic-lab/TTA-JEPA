from training.finetune import finetune
from evaluation.predict import evaluate


def run(args, config, grid):
    method = config.get("method", "tte_jepa")
    if method == "xgboost":
        from training.xgboost import fit
        from evaluation.xgboost import evaluate as evaluator
        if args.stage in ("pretrain", "tte"):
            raise ValueError("XGBoost has no representation pretraining; use finetune, run or evaluate")
        checkpoint = args.checkpoint or args.output_dir / "xgboost"
    else:
        evaluator = evaluate
        directory = "tte_training" if method == "icarefm" else "pretraining"
        checkpoint = args.checkpoint or args.output_dir / directory / "best.pt"
    if args.stage == "evaluate":
        if method != "xgboost":
            checkpoint = args.checkpoint or args.output_dir / "finetuning/best.pt"
        evaluator(grid(args.split), config, args.output_dir / "evaluation", checkpoint, args.device)
        return
    train, val = grid("train"), grid("val")
    if train.dimensions != val.dimensions or train.schema_key != val.schema_key:
        raise ValueError("Train/validation schema differs")
    if set(map(str, train.ids)) & set(map(str, val.ids)):
        raise ValueError("Admission overlap between train and validation")
    if method == "xgboost":
        checkpoint = fit(train, val, config, checkpoint, args.resume)
    else:
        if method == "tte_jepa":
            from training.pretrain import pretrain
        else:
            from training.baseline import pretrain
        if args.stage in ("run", "pretrain"):
            checkpoint = pretrain(train, val, config, args.output_dir / "pretraining", args.device, args.resume)
        if method == "icarefm" and args.stage in ("run", "tte"):
            if args.stage == "tte":
                checkpoint = args.checkpoint or args.output_dir / "pretraining/best.pt"
            checkpoint = pretrain(train, val, config, args.output_dir / "tte_training", args.device, args.resume,
                                  source=checkpoint, stage="tte_training")
        elif args.stage == "tte":
            raise ValueError("The separate TTE stage is only used by ICareFM")
        if args.stage in ("run", "finetune"):
            checkpoint = finetune(train, val, config, args.output_dir / "finetuning", checkpoint, args.device, args.resume)
    if args.stage == "run":
        test = grid("test")
        if (set(map(str, train.ids)) | set(map(str, val.ids))) & set(map(str, test.ids)):
            raise ValueError("Test admissions overlap training/validation")
        evaluator(test, config, args.output_dir / "evaluation", checkpoint, args.device)
