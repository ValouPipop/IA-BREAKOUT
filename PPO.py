"""
PPO (Proximal Policy Optimization) sur Atari Breakout
=====================================================
Évolution directe de A2C avec :
  1. Clipped surrogate objective → mises à jour stables
  2. Multiple passes sur les mêmes données (4 epochs par rollout)
  3. GAE (Generalized Advantage Estimation) → avantages plus précis
  4. Architecture IMPALA ResNet (même CNN que A2C)
  5. Collecte par rollouts de 128 steps (pas par épisode)

Pourquoi PPO > A2C ?
  → A2C utilise chaque expérience UNE SEULE FOIS puis la jette.
  → PPO réutilise les mêmes données 4 fois, avec un ratio clippé
    qui empêche les poids de trop changer → plus stable ET plus efficace.
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans écran (VM headless)
import matplotlib.pyplot as plt
from collections import deque
import cv2

gym.register_envs(ale_py)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
GAME         = "ALE/Breakout-v5"
K_FRAMES     = 4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95          # ★ Lambda pour GAE (lissage des avantages)
LR           = 2.5e-4        # LR standard PPO (plus bas que A2C)
CLIP_EPS     = 0.2           # ★ Zone de clipping [0.8, 1.2]
CV           = 0.5           # Coefficient perte critique
ENTROPY_COEF = 0.01          # Coefficient entropie (exploration)
GRAD_CLIP    = 0.5           # Gradient clipping
ROLLOUT_LEN  = 128           # ★ Steps collectés avant chaque update
PPO_EPOCHS   = 4             # ★ Passes sur les mêmes données
MINI_BATCH   = 64            # Taille mini-batch pour les updates PPO
N_TIMESTEPS  = 10_000_000    # ~20 000 épisodes selon la durée moyenne
PRINT_EVERY  = 20            # Affichage tous les N épisodes
SAVE_EVERY   = 500           # Checkpoint tous les N épisodes
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositif : {DEVICE}")
print(f"Algorithme : PPO")
print(f"Jeu : {GAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRÉTRAITEMENT VISUEL (identique à A2C)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, k: int = K_FRAMES):
        self.k      = k
        self.frames = deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(frame))
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.stack(list(self.frames), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ARCHITECTURE IMPALA ResNet (identique à A2C)
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(x)
        out = self.conv1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return out + residual


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class PPONet(nn.Module):
    """
    Réseau Actor-Critic avec backbone IMPALA ResNet.
    Architecture identique à A2CNet — seul l'algorithme d'entraînement change.
    """
    def __init__(self, n_actions: int, k_frames: int = K_FRAMES):
        super().__init__()

        channels = [32, 64, 64]
        self.conv_sequences = nn.ModuleList()
        in_ch = k_frames
        for out_ch in channels:
            self.conv_sequences.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        cnn_out_size = self._get_cnn_out(k_frames)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Fixup : zero-init la 2ème conv de chaque ResidualBlock
        for conv_seq in self.conv_sequences:
            nn.init.zeros_(conv_seq.res_block1.conv2.weight)
            nn.init.zeros_(conv_seq.res_block1.conv2.bias)
            nn.init.zeros_(conv_seq.res_block2.conv2.weight)
            nn.init.zeros_(conv_seq.res_block2.conv2.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=0.01)

    def _get_cnn_out(self, k_frames: int) -> int:
        dummy = torch.zeros(1, k_frames, 84, 84)
        for conv_seq in self.conv_sequences:
            dummy = conv_seq(dummy)
        dummy = torch.relu(dummy)
        return int(dummy.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor):
        for conv_seq in self.conv_sequences:
            x = conv_seq(x)
        x = torch.relu(x)
        x = x.reshape(x.size(0), -1)
        z      = self.fc(x)
        logits = self.actor(z)
        value  = self.critic(z).squeeze(-1)
        return logits, value


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GAE – Generalized Advantage Estimation
# ─────────────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    GAE(λ) : estimation des avantages plus précise que A2C.

    A2C :  A_t = R_t - V(s_t)         → haute variance
    GAE :  A_t = Σ (γλ)^l δ_{t+l}     → compromis biais/variance

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)   (TD error)

    λ=0 → pur TD (faible variance, haut biais)
    λ=1 → pur Monte-Carlo comme A2C (haute variance, pas de biais)
    λ=0.95 → bon compromis ★
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(n)):
        next_val = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MISE À JOUR PPO
# ─────────────────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, states_np, actions, old_log_probs,
               advantages, returns):
    """
    ★ Clipped Surrogate Objective :

    ratio = π_new(a|s) / π_old(a|s)
    L_clip = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)

    Si ratio sort de [0.8, 1.2], la loss est "clampée" → le gradient
    pousse MOINS fort → la politique ne change pas trop d'un coup.
    C'est ce qui rend PPO stable même avec 4 epochs sur les mêmes données.
    """
    n = len(actions)

    states_t    = torch.FloatTensor(states_np).to(DEVICE)
    actions_t   = torch.LongTensor(actions).to(DEVICE)
    old_lp_t    = torch.FloatTensor(old_log_probs).to(DEVICE)
    advs_t      = torch.FloatTensor(advantages).to(DEVICE)
    returns_t   = torch.FloatTensor(returns).to(DEVICE)

    # Normalisation des avantages
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    total_loss    = 0.0
    total_entropy = 0.0
    total_actor   = 0.0
    total_critic  = 0.0
    n_updates     = 0

    for epoch in range(PPO_EPOCHS):
        # ★ Shuffle les indices à chaque epoch pour varier les mini-batches
        indices = np.random.permutation(n)

        for start in range(0, n, MINI_BATCH):
            end    = min(start + MINI_BATCH, n)
            mb_idx = indices[start:end]

            mb_states   = states_t[mb_idx]
            mb_actions  = actions_t[mb_idx]
            mb_old_lp   = old_lp_t[mb_idx]
            mb_advs     = advs_t[mb_idx]
            mb_returns  = returns_t[mb_idx]

            # Forward pass
            logits, values = model(mb_states)
            dist      = torch.distributions.Categorical(logits=logits)
            new_lp    = dist.log_prob(mb_actions)
            entropy   = dist.entropy().mean()

            # ★ Ratio π_new / π_old
            ratio = torch.exp(new_lp - mb_old_lp)

            # ★ Clipped surrogate
            surr1 = ratio * mb_advs
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs

            L_actor  = -torch.min(surr1, surr2).mean()
            L_critic = ((mb_returns - values) ** 2).mean()

            loss = L_actor + CV * L_critic - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss    += loss.item()
            total_actor   += L_actor.item()
            total_critic  += L_critic.item()
            total_entropy += entropy.item()
            n_updates     += 1

    return {
        "loss":     total_loss / max(n_updates, 1),
        "L_actor":  total_actor / max(n_updates, 1),
        "L_critic": total_critic / max(n_updates, 1),
        "entropy":  total_entropy / max(n_updates, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BOUCLE D'ENTRAÎNEMENT PPO
# ─────────────────────────────────────────────────────────────────────────────
def train():
    env       = gym.make(GAME)
    n_actions = env.action_space.n
    stacker   = FrameStack(K_FRAMES)

    model     = PPONet(n_actions, K_FRAMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # LR annealing linéaire
    total_updates = N_TIMESTEPS // ROLLOUT_LEN
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_updates
    )

    print(f"Actions : {n_actions}")
    print(f"Total timesteps : {N_TIMESTEPS:,}")
    print(f"Rollout length : {ROLLOUT_LEN}")
    print(f"PPO epochs : {PPO_EPOCHS}")
    print(model)

    episode_rewards = []
    moving_avg      = []
    window          = deque(maxlen=100)
    best_avg        = 0.0

    # ── Initialisation de l'environnement ─────────────────────────────────
    obs, info  = env.reset()
    state      = stacker.reset(obs)
    obs, _, terminated, truncated, info = env.step(1)  # FIRE
    if not (terminated or truncated):
        state = stacker.step(obs)
    lives      = info.get("lives", 5)
    ep_reward  = 0.0
    ep_count   = 0
    total_steps = 0

    # ── Boucle principale ─────────────────────────────────────────────────
    while total_steps < N_TIMESTEPS:

        # ── Collecte d'un rollout de ROLLOUT_LEN steps ────────────────────
        rollout_states    = []
        rollout_actions   = []
        rollout_log_probs = []
        rollout_rewards   = []
        rollout_dones     = []
        rollout_values    = []

        for _ in range(ROLLOUT_LEN):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, value = model(state_t)

            dist     = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action).item()
            action   = action.item()
            val      = value.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            new_lives = info.get("lives", lives)
            life_lost = (new_lives < lives)
            lives     = new_lives

            clipped_reward = np.clip(reward, -1.0, 1.0)
            effective_done = float(done or life_lost)

            next_state = stacker.step(next_obs)

            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(clipped_reward)
            rollout_dones.append(effective_done)
            rollout_values.append(val)

            ep_reward   += reward
            state        = next_state
            total_steps += 1

            # Fire après perte de vie
            if life_lost and not done:
                obs, _, terminated, truncated, info = env.step(1)
                if not (terminated or truncated):
                    state = stacker.step(obs)
                else:
                    done = True

            # ── Fin d'épisode ─────────────────────────────────────────────
            if done:
                ep_count += 1
                episode_rewards.append(ep_reward)
                window.append(ep_reward)
                avg = np.mean(window)
                moving_avg.append(avg)

                if avg > best_avg:
                    best_avg = avg
                    torch.save(model.state_dict(), "ppo_breakout_best.pth")

                if ep_count % SAVE_EVERY == 0:
                    checkpoint = {
                        "episode": ep_count,
                        "timestep": total_steps,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_avg": best_avg,
                        "episode_rewards": episode_rewards,
                        "moving_avg": moving_avg,
                    }
                    torch.save(checkpoint, f"checkpoint_ppo_ep{ep_count}.pth")
                    print(f"  💾 Checkpoint : checkpoint_ppo_ep{ep_count}.pth")

                if ep_count % PRINT_EVERY == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Ep {ep_count:6d} | "
                          f"Steps {total_steps:>10,} / {N_TIMESTEPS:,} | "
                          f"Récomp : {ep_reward:5.1f} | "
                          f"Moy(100) : {avg:6.2f} | "
                          f"Best : {best_avg:6.2f} | "
                          f"LR : {lr:.2e}")

                # Reset environnement
                obs, info = env.reset()
                state     = stacker.reset(obs)
                obs, _, terminated, truncated, info = env.step(1)  # FIRE
                if not (terminated or truncated):
                    state = stacker.step(obs)
                lives     = info.get("lives", 5)
                ep_reward = 0.0

        # ── Bootstrap value pour GAE ──────────────────────────────────────
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            _, next_value = model(state_t)
            next_value = next_value.item()

        # ── Calcul GAE ────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones, next_value
        )

        # ── Mise à jour PPO (4 epochs sur les mêmes données) ─────────────
        result = ppo_update(
            model, optimizer,
            np.array(rollout_states), rollout_actions, rollout_log_probs,
            advantages, returns
        )

        scheduler.step()

    env.close()
    print(f"\n{'='*60}")
    print(f"Training terminé ! {ep_count} épisodes, {total_steps:,} steps")
    print(f"Meilleur avg(100) : {best_avg:.2f}")
    print(f"{'='*60}")
    return episode_rewards, moving_avg, model, ep_count


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(episode_rewards, moving_avg):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(episode_rewards, alpha=0.3, color="steelblue",
            label="Récompense par épisode")
    ax.plot(moving_avg, color="darkorange", linewidth=2,
            label="Moyenne glissante (100 épisodes)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.set_title("PPO – Breakout-v5 (IMPALA ResNet)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ppo_rewards.png", dpi=150)
    print("Figure sauvegardée sous 'ppo_rewards.png'")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    episode_rewards, moving_avg, model, ep_count = train()
    plot_results(episode_rewards, moving_avg)

    torch.save(model.state_dict(), "ppo_breakout_final.pth")
    print("Modèle final sauvegardé sous 'ppo_breakout_final.pth'")
    print("Meilleur modèle sauvegardé sous 'ppo_breakout_best.pth'")
