"""
PPO Vectorisé V2 sur Atari Breakout
====================================
VERSION AMÉLIORÉE : 16 envs, ROLLOUT 256, FC 512, 20M steps.

Améliorations par rapport à V1 (370 avg) :
  → 16 environnements (au lieu de 8) = 2x plus de données variées
  → ROLLOUT_LEN 256 (au lieu de 128) = trajectoires plus longues
  → FC 512 (au lieu de 256) = plus de capacité de décision
  → 20M timesteps (au lieu de 10M) = entraînement 2x plus long
  → Objectif : 500+ de score moyen

Architecture : IMPALA ResNet + PPO + GAE + 16 envs parallèles
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import cv2
import time

gym.register_envs(ale_py)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
GAME         = "ALE/Breakout-v5"
K_FRAMES     = 4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
LR           = 2.5e-4        # LR standard PPO
CLIP_EPS     = 0.2           # Clipping [0.8, 1.2]
CV           = 0.5           # Coeff perte critique
ENTROPY_COEF = 0.01          # Coeff entropie
GRAD_CLIP    = 0.5
N_ENVS       = 16            # ★ 16 environnements parallèles (2x plus de diversité)
ROLLOUT_LEN  = 256           # ★ Steps par env par rollout (total: 16×256 = 4096)
PPO_EPOCHS   = 4             # Passes sur les mêmes données
MINI_BATCH   = 512           # ★ Mini-batch (sur les 4096 transitions)
N_TIMESTEPS  = 20_000_000    # ★ 20M steps (2x plus long)
PRINT_EVERY  = 20
SAVE_EVERY   = 500
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositif : {DEVICE}")
print(f"Algorithme : PPO Vectorisé ({N_ENVS} envs)")
print(f"Jeu : {GAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRÉTRAITEMENT VISUEL
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
# 2.  ARCHITECTURE IMPALA ResNet
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = torch.relu(x)
        out = self.conv1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        return out + residual


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class PPONet(nn.Module):
    def __init__(self, n_actions, k_frames=K_FRAMES):
        super().__init__()
        channels = [32, 64, 64]
        self.conv_sequences = nn.ModuleList()
        in_ch = k_frames
        for out_ch in channels:
            self.conv_sequences.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        cnn_out_size = self._get_cnn_out(k_frames)
        self.fc = nn.Sequential(nn.Linear(cnn_out_size, 512), nn.ReLU())  # ★ 512 au lieu de 256
        self.actor  = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
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
        for conv_seq in self.conv_sequences:
            nn.init.zeros_(conv_seq.res_block1.conv2.weight)
            nn.init.zeros_(conv_seq.res_block1.conv2.bias)
            nn.init.zeros_(conv_seq.res_block2.conv2.weight)
            nn.init.zeros_(conv_seq.res_block2.conv2.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=0.01)

    def _get_cnn_out(self, k_frames):
        dummy = torch.zeros(1, k_frames, 84, 84)
        for conv_seq in self.conv_sequences:
            dummy = conv_seq(dummy)
        dummy = torch.relu(dummy)
        return int(dummy.reshape(1, -1).shape[1])

    def forward(self, x):
        for conv_seq in self.conv_sequences:
            x = conv_seq(x)
        x = torch.relu(x)
        x = x.reshape(x.size(0), -1)
        z      = self.fc(x)
        logits = self.actor(z)
        value  = self.critic(z).squeeze(-1)
        return logits, value


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ENVIRONNEMENT VECTORISÉ (géré manuellement pour fire-on-reset + lives)
# ─────────────────────────────────────────────────────────────────────────────
class VecBreakout:
    """
    Gère N_ENVS environnements Breakout en parallèle.
    Chaque env a son propre FrameStack, fire-on-reset, et life tracking.
    """
    def __init__(self, n_envs, game=GAME):
        self.n_envs  = n_envs
        self.envs    = [gym.make(game) for _ in range(n_envs)]
        self.stackers = [FrameStack(K_FRAMES) for _ in range(n_envs)]
        self.lives   = [5] * n_envs
        self.n_actions = self.envs[0].action_space.n

        # Initialiser tous les envs
        self.states = []
        for i in range(n_envs):
            state = self._reset_env(i)
            self.states.append(state)

    def _reset_env(self, idx):
        """Reset un env + fire + init stacker."""
        obs, info = self.envs[idx].reset()
        state = self.stackers[idx].reset(obs)

        # Fire pour lancer la balle
        obs, _, terminated, truncated, info = self.envs[idx].step(1)
        if not (terminated or truncated):
            state = self.stackers[idx].step(obs)

        self.lives[idx] = info.get("lives", 5)
        return state

    def step(self, actions):
        """
        Effectue un step sur tous les envs simultanément.
        Retourne : states, rewards, dones, life_losts, infos
        """
        next_states = []
        rewards     = []
        dones       = []
        life_losts  = []
        ep_rewards  = []  # Récompenses des épisodes terminés

        for i in range(self.n_envs):
            obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])
            done = terminated or truncated

            new_lives = info.get("lives", self.lives[i])
            life_lost = (new_lives < self.lives[i])
            self.lives[i] = new_lives

            clipped_reward = np.clip(reward, -1.0, 1.0)
            effective_done = float(done or life_lost)

            if done:
                # Épisode terminé → reset
                ep_rewards.append(info.get("episode", {}).get("r", reward))
                state = self._reset_env(i)
            else:
                state = self.stackers[i].step(obs)
                # Fire après perte de vie
                if life_lost:
                    obs2, _, term2, trunc2, info2 = self.envs[i].step(1)
                    if not (term2 or trunc2):
                        state = self.stackers[i].step(obs2)
                    else:
                        state = self._reset_env(i)
                        effective_done = 1.0

            next_states.append(state)
            rewards.append(clipped_reward)
            dones.append(effective_done)

            self.states[i] = state

        return (np.array(next_states), np.array(rewards),
                np.array(dones), ep_rewards)

    def get_states(self):
        return np.array(self.states)

    def close(self):
        for env in self.envs:
            env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GAE
# ─────────────────────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_values, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    GAE pour environnements vectorisés.
    rewards : (ROLLOUT_LEN, N_ENVS)
    values  : (ROLLOUT_LEN, N_ENVS)
    dones   : (ROLLOUT_LEN, N_ENVS)
    next_values : (N_ENVS,)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MISE À JOUR PPO
# ─────────────────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, states, actions, old_log_probs,
               advantages, returns):
    """
    PPO update avec mini-batches shufflés.
    Toutes les entrées sont déjà aplaties : (ROLLOUT_LEN × N_ENVS, ...)
    """
    n = len(actions)

    states_t    = torch.FloatTensor(states).to(DEVICE)
    actions_t   = torch.LongTensor(actions).to(DEVICE)
    old_lp_t    = torch.FloatTensor(old_log_probs).to(DEVICE)
    advs_t      = torch.FloatTensor(advantages).to(DEVICE)
    returns_t   = torch.FloatTensor(returns).to(DEVICE)

    # Normalisation des avantages
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    total_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for epoch in range(PPO_EPOCHS):
        indices = np.random.permutation(n)

        for start in range(0, n, MINI_BATCH):
            end    = min(start + MINI_BATCH, n)
            mb_idx = indices[start:end]

            mb_states   = states_t[mb_idx]
            mb_actions  = actions_t[mb_idx]
            mb_old_lp   = old_lp_t[mb_idx]
            mb_advs     = advs_t[mb_idx]
            mb_returns  = returns_t[mb_idx]

            logits, values = model(mb_states)
            dist      = torch.distributions.Categorical(logits=logits)
            new_lp    = dist.log_prob(mb_actions)
            entropy   = dist.entropy().mean()

            # ★ Clipped surrogate
            ratio = torch.exp(new_lp - mb_old_lp)
            surr1 = ratio * mb_advs
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs

            L_actor  = -torch.min(surr1, surr2).mean()
            L_critic = ((mb_returns - values) ** 2).mean()
            loss     = L_actor + CV * L_critic - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss    += loss.item()
            total_entropy += entropy.item()
            n_updates     += 1

    return {
        "loss":    total_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def train():
    vec_env   = VecBreakout(N_ENVS)
    n_actions = vec_env.n_actions

    model     = PPONet(n_actions, K_FRAMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    total_updates = N_TIMESTEPS // (ROLLOUT_LEN * N_ENVS)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_updates
    )

    print(f"Actions : {n_actions}")
    print(f"Envs parallèles : {N_ENVS}")
    print(f"Steps par rollout : {ROLLOUT_LEN} × {N_ENVS} = {ROLLOUT_LEN * N_ENVS}")
    print(f"Total timesteps : {N_TIMESTEPS:,}")
    print(f"Total updates : {total_updates:,}")
    print(model)

    episode_rewards = []
    moving_avg      = []
    window          = deque(maxlen=100)
    best_avg        = 0.0
    total_steps     = 0
    ep_count        = 0
    start_time      = time.time()

    # ── Tracking des récompenses brutes par env ───────────────────────────
    env_ep_rewards = [0.0] * N_ENVS

    while total_steps < N_TIMESTEPS:

        # ── Rollout : collecter ROLLOUT_LEN steps × N_ENVS ────────────────
        rollout_states    = []
        rollout_actions   = []
        rollout_log_probs = []
        rollout_rewards   = []
        rollout_dones     = []
        rollout_values    = []

        for t in range(ROLLOUT_LEN):
            states = vec_env.get_states()  # (N_ENVS, K, 84, 84)
            states_t = torch.FloatTensor(states).to(DEVICE)

            with torch.no_grad():
                logits, values = model(states_t)

            dist     = torch.distributions.Categorical(logits=logits)
            actions  = dist.sample()
            log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()

            next_states, rewards, dones, ep_infos = vec_env.step(actions_np)

            rollout_states.append(states)
            rollout_actions.append(actions_np)
            rollout_log_probs.append(log_probs.cpu().numpy())
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            rollout_values.append(values.cpu().numpy())

            # Tracking récompenses brutes
            for i in range(N_ENVS):
                env_ep_rewards[i] += rewards[i]
                if dones[i] >= 1.0:
                    # Fin d'épisode ou perte de vie → on ne compte que done=episode
                    pass

            # Compter les épisodes terminés
            for r in ep_infos:
                ep_count += 1
                # Récupérer la récompense réelle non-clippée serait idéal,
                # mais on utilise la somme des rewards clippés comme proxy
                pass

            total_steps += N_ENVS

        # ── Convertir en arrays (ROLLOUT_LEN, N_ENVS) ────────────────────
        rollout_states    = np.array(rollout_states)     # (T, N, K, 84, 84)
        rollout_actions   = np.array(rollout_actions)    # (T, N)
        rollout_log_probs = np.array(rollout_log_probs)  # (T, N)
        rollout_rewards   = np.array(rollout_rewards)    # (T, N)
        rollout_dones     = np.array(rollout_dones)      # (T, N)
        rollout_values    = np.array(rollout_values)     # (T, N)

        # ── Bootstrap value ───────────────────────────────────────────────
        with torch.no_grad():
            last_states = torch.FloatTensor(vec_env.get_states()).to(DEVICE)
            _, next_values = model(last_states)
            next_values = next_values.cpu().numpy()

        # ── GAE ───────────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones, next_values
        )

        # ── Aplatir (T, N, ...) → (T*N, ...) ─────────────────────────────
        T, N = ROLLOUT_LEN, N_ENVS
        flat_states    = rollout_states.reshape(T * N, K_FRAMES, 84, 84)
        flat_actions   = rollout_actions.reshape(T * N)
        flat_log_probs = rollout_log_probs.reshape(T * N)
        flat_advs      = advantages.reshape(T * N)
        flat_returns   = returns.reshape(T * N)

        # ── PPO Update ────────────────────────────────────────────────────
        result = ppo_update(
            model, optimizer,
            flat_states, flat_actions, flat_log_probs,
            flat_advs, flat_returns
        )

        scheduler.step()

        # ── Suivi des épisodes ────────────────────────────────────────────
        # Évaluation rapide toutes les N updates : jouer 1 épisode complet
        n_update = total_steps // (ROLLOUT_LEN * N_ENVS)
        if n_update % 10 == 0:
            eval_reward = evaluate_episode(model)
            episode_rewards.append(eval_reward)
            window.append(eval_reward)
            avg = np.mean(window)
            moving_avg.append(avg)

            if avg > best_avg:
                best_avg = avg
                torch.save(model.state_dict(), "ppo_breakout_best.pth")

            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            lr = optimizer.param_groups[0]['lr']
            print(f"Steps {total_steps:>10,} / {N_TIMESTEPS:,} | "
                  f"Eval : {eval_reward:5.1f} | "
                  f"Moy(100) : {avg:6.2f} | "
                  f"Best : {best_avg:6.2f} | "
                  f"FPS : {fps:,.0f} | "
                  f"Entropie : {result['entropy']:.3f} | "
                  f"LR : {lr:.2e}")

            if len(episode_rewards) % SAVE_EVERY == 0 and len(episode_rewards) > 0:
                checkpoint = {
                    "timestep": total_steps,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_avg": best_avg,
                    "episode_rewards": episode_rewards,
                    "moving_avg": moving_avg,
                }
                torch.save(checkpoint, f"checkpoint_ppo_{total_steps}.pth")
                print(f"  💾 Checkpoint sauvegardé")

    vec_env.close()
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training terminé ! {total_steps:,} steps en {elapsed/3600:.1f}h")
    print(f"Meilleur avg(100) : {best_avg:.2f}")
    print(f"{'='*60}")
    return episode_rewards, moving_avg, model


def evaluate_episode(model):
    """Joue UN épisode complet pour évaluer le modèle (sans exploration)."""
    env = gym.make(GAME)
    stacker = FrameStack(K_FRAMES)

    obs, info = env.reset()
    state = stacker.reset(obs)
    obs, _, terminated, truncated, info = env.step(1)
    if not (terminated or truncated):
        state = stacker.step(obs)
    lives = info.get("lives", 5)

    total_reward = 0.0
    done = False

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(state_t)
        # ★ Greedy (argmax) pour l'évaluation, pas de sampling
        action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        new_lives = info.get("lives", lives)
        life_lost = (new_lives < lives)
        lives = new_lives

        state = stacker.step(obs)

        if life_lost and not done:
            obs, _, terminated, truncated, info = env.step(1)
            if not (terminated or truncated):
                state = stacker.step(obs)
            else:
                done = True

    env.close()
    return total_reward


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(episode_rewards, moving_avg):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(episode_rewards, alpha=0.3, color="steelblue",
            label="Évaluation par épisode")
    ax.plot(moving_avg, color="darkorange", linewidth=2,
            label="Moyenne glissante (100)")
    ax.set_xlabel("Évaluation")
    ax.set_ylabel("Récompense totale")
    ax.set_title("PPO Vectorisé V2 – Breakout-v5 (16 envs × IMPALA ResNet 512)")
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
    episode_rewards, moving_avg, model = train()
    plot_results(episode_rewards, moving_avg)

    torch.save(model.state_dict(), "ppo_breakout_final.pth")
    print("Modèle final sauvegardé sous 'ppo_breakout_final.pth'")
    print("Meilleur modèle sauvegardé sous 'ppo_breakout_best.pth'")
