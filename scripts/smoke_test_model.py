
import torch

from mini_transformer.config import (
    load_config,
    resolve_model_vocab_size,
)
from mini_transformer.dataset import (
    LanguageModelDataset,
    load_token_ids,
)
from mini_transformer.model import GPTLanguageModel
from mini_transformer.tokenizer import CharacterTokenizer


def main():
    torch.manual_seed(42)

    app_config = load_config("configs/baseline.yaml")

    tokenizer = CharacterTokenizer.load(
        "data/processed/tokenizer.json"
    )

    model_config = resolve_model_vocab_size(
        app_config.model,
        tokenizer.vocab_size,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = GPTLanguageModel(model_config).to(device)

    train_tokens = load_token_ids(
        app_config.data.train_path
    )

    dataset = LanguageModelDataset(
        train_tokens,
        block_size=model_config.block_size,
    )

    x, y = dataset.get_batch(
        batch_size=app_config.training.batch_size,
        device=device,
    )

    logits, loss = model(x, y)

    print(f"Device: {device}")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Target shape: {tuple(y.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.4f}")

    assert logits.shape == (
        app_config.training.batch_size,
        model_config.block_size,
        tokenizer.vocab_size,
    )

    assert loss is not None
    assert torch.isfinite(loss)

    # Verify that the entire model can participate in backpropagation.
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)

    print("Forward pass: successful")
    print("Backward pass: successful")


if __name__ == "__main__":
    main()
