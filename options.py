import argparse


def parse_options():
    parser = argparse.ArgumentParser(description="Gene-Conditioned Context Retrieval")
    parser.add_argument(
        "command", choices=["demo", "context", "extract", "train", "evaluate", "predict"]
    )
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--genes", default="data/genes.csv")
    parser.add_argument("--queries", default="data/gene_queries.npy")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="DeepSpot-M model.safetensors for target-decoder initialization",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Official DeepSpot-M/Midnight model directory for feature extraction",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-genes", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--patience", type=int, default=0, help="Validation PCC patience; 0 disables early stopping"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()
