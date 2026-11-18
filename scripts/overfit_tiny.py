
from __future__ import annotations

import torch

from mini_transformer.config import ModelConfig, TrainingConfig
from mini_transformer.dataset import LanguageModelDataset
from mini_transformer.model import GPTLanguageModel
from mini_transformer.training import create_optimizer


def main():
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # A deliberately small, repetitive sequence.
    tokens = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 8] * 16,
        dtype=torch.long,
    )

    block_size = 8

    dataset = LanguageModelDataset(
        tokens,
        block_size=block_size,
    )

    model_config = ModelConfig(
        vocab_size=9,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )

    model = GPTLanguageModel(model_config).to(device)

    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=0.003,
        weight_decay=0.0,
        max_steps=1000,
        eval_interval=100,
        gradient_clip=1.0,
        warmup_steps=0,
        min_learning_rate=0.0003,
        eval_batches=10,
        checkpoint_interval=1000,
    )

    optimizer = create_optimizer(model, training_config)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    model.train()

    for step in range(training_config.max_steps):
        x, y = dataset.get_batch(
            batch_size=training_config.batch_size,
            device=device,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected training loss.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            training_config.gradient_clip,
        )

        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f"step {step + 1:4d} | loss {loss.item():.6f}"
            )

    model.eval()

    x, y = dataset.get_example(0)

    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits, final_loss = model(x, y)

    print("\nFinal example:")
    print("Input:", x)
    print("Target:", y)
    print("Logits shape:", logits.shape)
    print("Final loss:", final_loss.item())


if __name__ == "__main__":
    main()
