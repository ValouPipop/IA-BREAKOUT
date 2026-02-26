"""
A2C Multi-Env avec IMPALA ResNet sur Breakout
===============================================
Le MEILLEUR des deux mondes :
  - A2C en mode EPISODE (update à la fin de chaque partie) → ça marchait ✅
  - 8 environnements en parallèle → 8x plus de données variées ✅
  - IMPALA ResNet (15 couches CNN) → extraction de features riche ✅
  - Gradient accumulation → pas d'OOM ✅

Chaque env joue sa partie indépendamment. Quand UNE env finit sa partie,
on fait une update A2C sur cette trajectoire complète, puis on reset cette env.
Les 7 autres envs continuent de jouer sans interruption.
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
LR           = 7e-4              # Même LR que ton A2C qui marchait
CV           = 0.5
ENTROPY_COEF = 0.01
GRAD_CLIP    = 0.5
N_ENVS       = 8                 # ★ 8 environnements parallèles
N_EPISODES   = 20000             # ★ Total épisodes (répartis sur les 8 envs)
MAX_BATCH    = 256               # Taille max mini-batch (gradient accumulation)
PRINT_EVERY  = 50
SAVE_EVERY   = 1000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositif : {DEVICE}")
print(f"Algorithme : A2C Multi-Env (EPISODE mode)")
print(f"Envs parallèles : {N_ENVS}")
print(f"Jeu : {GAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRÉTRAITEMENT VISUEL
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, k=K_FRAMES):
        self.k      = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame):
        self.frames.append(preprocess_frame(frame))
        return self._get_state()

    def _get_state(self):
        return np.stack(list(self.frames), axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ARCHITECTURE IMPALA ResNet
# ─────────────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels):
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


class A2CNet(nn.Module):
    """
    Réseau Actor-Critic avec backbone IMPALA ResNet.
    Identique à celui de CA2C.py — le même qui faisait 260 avg.
    """
    def __init__(self, n_actions, k_frames=K_FRAMES):
        super().__init__()
        channels = [32, 64, 64]
        self.conv_sequences = nn.ModuleList()
        in_ch = k_frames
        for out_ch in channels:
            self.conv_sequences.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        cnn_out = self._get_cnn_out(k_frames)
        self.fc = nn.Sequential(nn.Linear(cnn_out, 256), nn.ReLU())
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
        for conv_seq in self.conv_sequences:
            nn.init.zeros_(conv_seq.res_block1.conv2.weight)
            nn.init.zeros_(conv_seq.res_block1.conv2.bias)
            nn.init.zeros_(conv_seq.res_block2.conv2.weight)
            nn.init.zeros_(conv_seq.res_block2.conv2.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=0.01)

    def _get_cnn_out(self, k_frames):
        dummy = torch.zeros(1, k_frames, 84, 84)
        for cs in self.conv_sequences:
            dummy = cs(dummy)
        return int(torch.relu(dummy).reshape(1, -1).shape[1])

    def forward(self, x):
        for cs in self.conv_sequences:
            x = cs(x)
        x = torch.relu(x)
        x = x.reshape(x.size(0), -1)
        z = self.fc(x)
        return self.actor(z), self.critic(z).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALCUL DES RETOURS
# ─────────────────────────────────────────────────────────────────────────────
def compute_returns(rewards, dones, gamma=GAMMA):
    T = len(rewards)
    returns = [0.0] * T
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * (1.0 - dones[t]) * g
        returns[t] = g
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MISE À JOUR A2C (avec gradient accumulation)
# ─────────────────────────────────────────────────────────────────────────────
def update_model(model, optimizer, states, actions, rewards, dones):
    """
    Même update A2C que CA2C.py avec gradient accumulation.
    C'est cette fonction qui donnait 260 avg.
    """
    if len(rewards) < 2:
        return None

    returns_np = compute_returns(rewards, dones)
    returns_t  = torch.FloatTensor(returns_np).to(DEVICE)
    states_np  = np.array(states)
    actions_t  = torch.LongTensor(actions).to(DEVICE)

    n = len(states)
    n_chunks = (n + MAX_BATCH - 1) // MAX_BATCH

    # Passe 1 : Forward sans gradient pour les avantages
    with torch.no_grad():
        values_list = []
        for i in range(0, n, MAX_BATCH):
            chunk = torch.FloatTensor(states_np[i:i+MAX_BATCH]).to(DEVICE)
            _, v = model(chunk)
            values_list.append(v)
        values_all = torch.cat(values_list, dim=0)

    advantages = returns_t - values_all
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Passe 2 : Gradient accumulation
    optimizer.zero_grad()
    total_entropy = 0.0

    for i in range(0, n, MAX_BATCH):
        j = min(i + MAX_BATCH, n)
        chunk_s = torch.FloatTensor(states_np[i:j]).to(DEVICE)
        chunk_a = actions_t[i:j]
        chunk_r = returns_t[i:j]
        chunk_adv = advantages[i:j]

        logits, values = model(chunk_s)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(chunk_a)

        L_actor   = -(log_probs * chunk_adv.detach()).mean()
        L_critic  = ((chunk_r - values) ** 2).mean()
        L_entropy = dist.entropy().mean()

        loss = (L_actor + CV * L_critic - ENTROPY_COEF * L_entropy) / n_chunks
        loss.backward()
        total_entropy += L_entropy.item()

    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {"entropy": total_entropy / n_chunks, "seg_len": n}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BOUCLE D'ENTRAÎNEMENT MULTI-ENV
# ─────────────────────────────────────────────────────────────────────────────
def train():
    # ── Créer les N_ENVS environnements ───────────────────────────────────
    envs     = [gym.make(GAME) for _ in range(N_ENVS)]
    n_actions = envs[0].action_space.n
    stackers = [FrameStack(K_FRAMES) for _ in range(N_ENVS)]

    model     = A2CNet(n_actions, K_FRAMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=N_EPISODES
    )

    print(f"Actions : {n_actions}")
    print(f"Épisodes cibles : {N_EPISODES}")
    print(model)

    # ── État de chaque env ────────────────────────────────────────────────
    states      = [None] * N_ENVS
    lives       = [5] * N_ENVS
    ep_rewards  = [0.0] * N_ENVS  # Récompense courante par env

    # Buffers par env (chaque env accumule SA trajectoire)
    buffers = [{"states": [], "actions": [], "rewards": [], "dones": []}
               for _ in range(N_ENVS)]

    # Initialiser chaque env
    for i in range(N_ENVS):
        obs, info = envs[i].reset()
        states[i] = stackers[i].reset(obs)
        obs, _, term, trunc, info = envs[i].step(1)  # FIRE
        if not (term or trunc):
            states[i] = stackers[i].step(obs)
        lives[i] = info.get("lives", 5)

    # ── Métriques globales ────────────────────────────────────────────────
    all_rewards = []
    moving_avg  = []
    window      = deque(maxlen=100)
    best_avg    = 0.0
    ep_count    = 0
    total_steps = 0
    start_time  = time.time()

    # ── Boucle principale ─────────────────────────────────────────────────
    while ep_count < N_EPISODES:

        # ── Préparer les états des N_ENVS ─────────────────────────────────
        batch_states = np.array(states)  # (N_ENVS, K, 84, 84)
        states_t = torch.FloatTensor(batch_states).to(DEVICE)

        # ── Forward pass pour les N_ENVS en UN SEUL batch ─────────────────
        with torch.no_grad():
            logits, _ = model(states_t)

        dist    = torch.distributions.Categorical(logits=logits)
        actions = dist.sample().cpu().numpy()

        # ── Step dans chaque env ──────────────────────────────────────────
        for i in range(N_ENVS):
            obs, reward, terminated, truncated, info = envs[i].step(actions[i])
            done = terminated or truncated

            new_lives = info.get("lives", lives[i])
            life_lost = (new_lives < lives[i])
            lives[i]  = new_lives

            clipped_reward = np.clip(reward, -1.0, 1.0)
            effective_done = float(done or life_lost)

            next_state = stackers[i].step(obs)

            # Ajouter au buffer de cet env
            buffers[i]["states"].append(states[i])
            buffers[i]["actions"].append(actions[i])
            buffers[i]["rewards"].append(clipped_reward)
            buffers[i]["dones"].append(effective_done)

            ep_rewards[i] += reward  # Récompense brute
            states[i] = next_state
            total_steps += 1

            # Fire après perte de vie
            if life_lost and not done:
                obs2, _, t2, tr2, info2 = envs[i].step(1)
                if not (t2 or tr2):
                    states[i] = stackers[i].step(obs2)
                else:
                    done = True

            # ── FIN D'ÉPISODE → UPDATE A2C ────────────────────────────────
            if done:
                ep_count += 1

                # ★ Update A2C sur la trajectoire COMPLÈTE de cet env
                result = update_model(
                    model, optimizer,
                    buffers[i]["states"],
                    buffers[i]["actions"],
                    buffers[i]["rewards"],
                    buffers[i]["dones"]
                )

                scheduler.step()

                # Tracking
                all_rewards.append(ep_rewards[i])
                window.append(ep_rewards[i])
                avg = np.mean(window)
                moving_avg.append(avg)

                if avg > best_avg:
                    best_avg = avg
                    torch.save(model.state_dict(), "a2c_multi_best.pth")

                if ep_count % PRINT_EVERY == 0:
                    elapsed = time.time() - start_time
                    fps = total_steps / elapsed
                    lr = optimizer.param_groups[0]['lr']
                    ent = result["entropy"] if result else 0
                    print(f"Ep {ep_count:6d}/{N_EPISODES} | "
                          f"Récomp : {ep_rewards[i]:5.1f} | "
                          f"Moy(100) : {avg:6.2f} | "
                          f"Best : {best_avg:6.2f} | "
                          f"Entropie : {ent:.3f} | "
                          f"FPS : {fps:,.0f} | "
                          f"LR : {lr:.2e}")

                if ep_count % SAVE_EVERY == 0:
                    checkpoint = {
                        "episode": ep_count,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_avg": best_avg,
                        "episode_rewards": all_rewards,
                        "moving_avg": moving_avg,
                    }
                    torch.save(checkpoint, f"checkpoint_multi_ep{ep_count}.pth")
                    print(f"  💾 Checkpoint : checkpoint_multi_ep{ep_count}.pth")

                # ── Reset cet env ─────────────────────────────────────────
                buffers[i] = {"states": [], "actions": [], "rewards": [], "dones": []}
                ep_rewards[i] = 0.0
                obs, info = envs[i].reset()
                states[i] = stackers[i].reset(obs)
                obs, _, t2, tr2, info = envs[i].step(1)
                if not (t2 or tr2):
                    states[i] = stackers[i].step(obs)
                lives[i] = info.get("lives", 5)

                if ep_count >= N_EPISODES:
                    break

    # ── Fin ───────────────────────────────────────────────────────────────
    for env in envs:
        env.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training terminé ! {ep_count} épisodes en {elapsed/3600:.1f}h")
    print(f"Meilleur avg(100) : {best_avg:.2f}")
    print(f"FPS moyen : {total_steps/elapsed:,.0f}")
    print(f"{'='*60}")
    return all_rewards, moving_avg, model


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
    ax.set_title(f"A2C Multi-Env ({N_ENVS} envs) – Breakout-v5 (IMPALA ResNet)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("a2c_multi_rewards.png", dpi=150)
    print("Figure sauvegardée sous 'a2c_multi_rewards.png'")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    episode_rewards, moving_avg, model = train()
    plot_results(episode_rewards, moving_avg)

    torch.save(model.state_dict(), "a2c_multi_final.pth")
    print("Modèle final sauvegardé sous 'a2c_multi_final.pth'")
    print("Meilleur modèle sauvegardé sous 'a2c_multi_best.pth'")
