import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.torch_layers import create_mlp


def parse_net_arch(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="booster_dataset/imitation_learning/bc_commands.npz",
        help="Path to observations/actions dataset.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training_runs/bc_actor.pt",
        help="Where to save the BC actor weights.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--net-arch", type=str, default="256,256,128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset)
    observations = np.array(data["observations"], dtype=np.float32)
    actions = np.array(data["actions"], dtype=np.float32)

    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    net_arch = parse_net_arch(args.net_arch)

    policy_net = nn.Sequential(
        *create_mlp(
            input_dim=obs_dim,
            output_dim=act_dim,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            squash_output=True,
        )
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(observations), torch.from_numpy(actions)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            pred = policy_net(batch_obs)
            loss = loss_fn(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:03d} | loss={avg_loss:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy_net.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "net_arch": net_arch,
        },
        out_path,
    )
    print(f"Saved BC actor to {out_path}")


if __name__ == "__main__":
    main()
