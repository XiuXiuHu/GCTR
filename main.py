import json

from options import parse_options


def main():
    options = parse_options()
    if options.command == "demo":
        import torch
        from engine import seed_everything
        from metrics import masked_mse
        from model import GCTR

        seed_everything(options.seed, options.threads)
        queries = torch.randn(options.num_genes, options.dim)
        model = GCTR(queries, options.heads, options.layers, options.dropout)
        target = torch.randn(2, options.tokens, options.dim)
        context = torch.randn(2, 80, options.dim)
        mask = torch.ones(2, 80, dtype=torch.bool)
        mask[1] = False
        prediction = model(target, context, mask)
        labels = torch.randn_like(prediction)
        supervision = torch.ones_like(prediction, dtype=torch.bool)
        supervision[0, 0] = False
        labels[~supervision] = float("nan")
        loss = masked_mse(prediction, labels, supervision)
        loss.backward()
        print(
            json.dumps(
                dict(
                    mode="synthetic_shape_check",
                    prediction_shape=list(prediction.shape),
                    parameters=sum(p.numel() for p in model.parameters()),
                    finite=bool(torch.isfinite(prediction).all()),
                    backward=True,
                    benchmark_evaluation=False,
                )
            )
        )
    elif options.command == "context":
        from data import build_context

        build_context(options.manifest)
    elif options.command == "extract":
        from features import extract_features

        extract_features(options)
    elif options.command == "train":
        from engine import train

        train(options)
    else:
        from engine import evaluate

        evaluate(options, labels=options.command == "evaluate")


if __name__ == "__main__":
    main()
