"""
Comparaison des deux modes de mise à jour :
  - Mode "EPISODE" : 1 update à la fin de chaque partie
  - Mode "LIFE"    : 1 update à chaque perte de vie

Entraîne les deux modèles séquentiellement puis affiche les courbes.
★ Version améliorée avec suivi détaillé de toutes les métriques.
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from CA2C import (
    A2CNet, FrameStack, compute_returns, update_model,
    GAME, K_FRAMES, GAMMA, LR, CV, ENTROPY_COEF, GRAD_CLIP, DEVICE
)

# ── Enregistrement des environnements Atari ──────────────────────────────────
gym.register_envs(ale_py)

# ── Paramètres de comparaison ─────────────────────────────────────────────────
N_EPISODES  = 5000       # Nombre d'épisodes par mode
PRINT_EVERY = 50         # Affichage tous les N épisodes


def train_with_mode(mode: str):
    """
    Entraîne un modèle A2C avec le mode spécifié.
    mode : "EPISODE" ou "LIFE"
    Retourne un dict avec toutes les métriques collectées.
    """
    assert mode in ("EPISODE", "LIFE"), f"Mode inconnu : {mode}"

    print(f"\n{'='*80}")
    print(f"  ENTRAÎNEMENT MODE : {mode}")
    print(f"{'='*80}\n")

    env       = gym.make(GAME)
    n_actions = env.action_space.n
    stacker   = FrameStack(K_FRAMES)

    model     = A2CNet(n_actions, K_FRAMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=N_EPISODES
    )

    # ── Métriques à collecter ─────────────────────────────────────────────
    episode_rewards = []       # Récompense brute par épisode
    moving_avg      = []       # Moyenne glissante (100 épisodes)
    all_losses      = []       # Loss totale moyenne par épisode
    all_actor_losses  = []     # Loss acteur moyenne par épisode
    all_critic_losses = []     # Loss critique moyenne par épisode
    all_entropies   = []       # Entropie moyenne par épisode
    all_steps       = []       # Nombre de steps par épisode
    all_updates     = []       # Nombre d'updates par épisode
    all_seg_lens    = []       # Longueur moyenne des segments par épisode
    all_lrs         = []       # Learning rate par épisode
    all_max_scores  = []       # Meilleur score atteint jusque-là

    window    = deque(maxlen=100)
    best_avg  = 0.0
    max_score = 0.0
    start_time = time.time()

    # ── GPU memory tracking ───────────────────────────────────────────────
    gpu_available = DEVICE.type == "cuda"

    for ep in range(1, N_EPISODES + 1):

        obs, info = env.reset()
        state     = stacker.reset(obs)

        # Fire-on-reset
        obs, _, terminated, truncated, info = env.step(1)
        if not (terminated or truncated):
            state = stacker.step(obs)

        lives = info.get("lives", 5)

        seg_states  = []
        seg_actions = []
        seg_rewards = []
        seg_dones   = []

        ep_reward    = 0.0
        done         = False
        ep_steps     = 0
        n_updates    = 0

        # Métriques d'update pour cet épisode
        ep_losses       = []
        ep_actor_losses = []
        ep_critic_losses = []
        ep_entropies    = []
        ep_seg_lens     = []

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, _ = model(state_t)

            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            new_lives = info.get("lives", lives)
            life_lost = (new_lives < lives)
            lives     = new_lives

            clipped_reward = np.clip(reward, -1.0, 1.0)
            effective_done = float(done or life_lost)

            next_state = stacker.step(next_obs)

            seg_states.append(state)
            seg_actions.append(action)
            seg_rewards.append(clipped_reward)
            seg_dones.append(effective_done)

            ep_reward += reward
            ep_steps  += 1
            state      = next_state

            # ── Mode LIFE : update à chaque perte de vie ──────────────────
            if mode == "LIFE" and life_lost and not done:
                result = update_model(
                    model, optimizer,
                    seg_states, seg_actions, seg_rewards, seg_dones
                )
                if result is not None:
                    ep_losses.append(result["loss"])
                    ep_actor_losses.append(result["L_actor"])
                    ep_critic_losses.append(result["L_critic"])
                    ep_entropies.append(result["entropy"])
                    ep_seg_lens.append(result["seg_len"])
                    n_updates += 1
                seg_states  = []
                seg_actions = []
                seg_rewards = []
                seg_dones   = []

            # Fire après perte de vie
            if life_lost and not done:
                obs, _, terminated, truncated, info = env.step(1)
                if not (terminated or truncated):
                    state = stacker.step(obs)
                else:
                    done = True

        # ── Update sur le segment restant (fin d'épisode) ─────────────────
        if len(seg_rewards) >= 2:
            result = update_model(
                model, optimizer,
                seg_states, seg_actions, seg_rewards, seg_dones
            )
            if result is not None:
                ep_losses.append(result["loss"])
                ep_actor_losses.append(result["L_actor"])
                ep_critic_losses.append(result["L_critic"])
                ep_entropies.append(result["entropy"])
                ep_seg_lens.append(result["seg_len"])
                n_updates += 1

        scheduler.step()

        # ── Collecte des métriques ────────────────────────────────────────
        episode_rewards.append(ep_reward)
        window.append(ep_reward)
        avg = np.mean(window)
        moving_avg.append(avg)

        max_score = max(max_score, ep_reward)
        all_max_scores.append(max_score)

        if avg > best_avg:
            best_avg = avg
            torch.save(model.state_dict(), f"a2c_best_{mode.lower()}.pth")

        # Moyennes des métriques de cet épisode
        all_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        all_actor_losses.append(np.mean(ep_actor_losses) if ep_actor_losses else 0.0)
        all_critic_losses.append(np.mean(ep_critic_losses) if ep_critic_losses else 0.0)
        all_entropies.append(np.mean(ep_entropies) if ep_entropies else 0.0)
        all_steps.append(ep_steps)
        all_updates.append(n_updates)
        all_seg_lens.append(np.mean(ep_seg_lens) if ep_seg_lens else 0.0)
        all_lrs.append(optimizer.param_groups[0]['lr'])

        # ── Affichage détaillé ────────────────────────────────────────────
        if ep % PRINT_EVERY == 0:
            elapsed     = time.time() - start_time
            eps_per_sec = ep / elapsed
            eta         = (N_EPISODES - ep) / eps_per_sec

            # GPU memory
            gpu_info = ""
            if gpu_available:
                gpu_alloc = torch.cuda.memory_allocated() / 1024**3
                gpu_reserv = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f" | GPU: {gpu_alloc:.2f}/{gpu_reserv:.2f} Go"

            current_lr = optimizer.param_groups[0]['lr']
            avg_steps  = np.mean(all_steps[-PRINT_EVERY:])
            avg_upd    = np.mean(all_updates[-PRINT_EVERY:])
            avg_seg    = np.mean(all_seg_lens[-PRINT_EVERY:])
            avg_loss   = np.mean(all_losses[-PRINT_EVERY:]) if any(all_losses[-PRINT_EVERY:]) else 0.0
            avg_la     = np.mean(all_actor_losses[-PRINT_EVERY:]) if any(all_actor_losses[-PRINT_EVERY:]) else 0.0
            avg_lc     = np.mean(all_critic_losses[-PRINT_EVERY:]) if any(all_critic_losses[-PRINT_EVERY:]) else 0.0
            avg_ent    = np.mean(all_entropies[-PRINT_EVERY:]) if any(all_entropies[-PRINT_EVERY:]) else 0.0

            print(f"  [{mode:7s}] Ép {ep:4d}/{N_EPISODES}"
                  f" | Réc: {ep_reward:5.1f}"
                  f" | Moy100: {avg:6.2f}"
                  f" | Best: {best_avg:6.2f}"
                  f" | Max: {max_score:5.0f}")
            print(f"           "
                  f" | Steps: {avg_steps:5.0f}"
                  f" | Upd: {avg_upd:3.1f}"
                  f" | SegLen: {avg_seg:5.0f}"
                  f" | LR: {current_lr:.2e}"
                  f" | ETA: {eta/60:.0f}min")
            print(f"           "
                  f" | Loss: {avg_loss:7.3f}"
                  f" | L_act: {avg_la:7.3f}"
                  f" | L_crit: {avg_lc:7.3f}"
                  f" | H: {avg_ent:.3f}"
                  f"{gpu_info}")
            print()

    env.close()

    elapsed_total = time.time() - start_time
    print(f"\n  [{mode}] Terminé en {elapsed_total/60:.1f} minutes")
    print(f"  [{mode}] Meilleure moyenne (100 ép.) : {best_avg:.2f}")
    print(f"  [{mode}] Score max atteint : {max_score:.0f}")
    print(f"  [{mode}] Steps moyen/épisode : {np.mean(all_steps):.0f}")
    print(f"  [{mode}] Updates moyen/épisode : {np.mean(all_updates):.1f}")

    torch.save(model.state_dict(), f"a2c_final_{mode.lower()}.pth")

    return {
        "rewards": episode_rewards,
        "moving_avg": moving_avg,
        "losses": all_losses,
        "actor_losses": all_actor_losses,
        "critic_losses": all_critic_losses,
        "entropies": all_entropies,
        "steps": all_steps,
        "updates": all_updates,
        "seg_lens": all_seg_lens,
        "lrs": all_lrs,
        "max_scores": all_max_scores,
        "best_avg": best_avg,
        "max_score": max_score,
        "total_time": elapsed_total,
    }


def smooth(data, window=100):
    """Moyenne glissante pour lisser les courbes."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid').tolist()


def plot_comparison(results: dict):
    """Affiche les courbes de comparaison détaillées."""

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle("Comparaison LIFE vs EPISODE – Métriques détaillées", fontsize=16, fontweight='bold')

    colors = {
        "EPISODE": {"raw": "lightcoral",   "avg": "red"},
        "LIFE":    {"raw": "lightskyblue", "avg": "dodgerblue"},
    }

    # ── 1. Récompenses ────────────────────────────────────────────────────
    ax = axes[0, 0]
    for mode, data in results.items():
        c = colors[mode]
        ax.plot(data["rewards"], alpha=0.1, color=c["raw"])
        ax.plot(data["moving_avg"], color=c["avg"], linewidth=2, label=f"{mode} (moy 100)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense")
    ax.set_title("Récompenses par épisode")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ── 2. Meilleur score atteint ─────────────────────────────────────────
    ax = axes[0, 1]
    for mode, data in results.items():
        c = colors[mode]
        ax.plot(data["max_scores"], color=c["avg"], linewidth=2, label=f"{mode} (max cumulé)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Score max atteint")
    ax.set_title("Progression du meilleur score")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ── 3. Loss totale ────────────────────────────────────────────────────
    ax = axes[1, 0]
    for mode, data in results.items():
        c = colors[mode]
        smoothed = smooth(data["losses"])
        ax.plot(smoothed, color=c["avg"], linewidth=1.5, label=f"{mode}", alpha=0.8)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Loss totale")
    ax.set_title("Loss totale (lissée sur 100 ép.)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ── 4. Entropie ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    for mode, data in results.items():
        c = colors[mode]
        smoothed = smooth(data["entropies"])
        ax.plot(smoothed, color=c["avg"], linewidth=1.5, label=f"{mode}", alpha=0.8)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Entropie")
    ax.set_title("Entropie de la politique (exploration)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ── 5. Steps par épisode ──────────────────────────────────────────────
    ax = axes[2, 0]
    for mode, data in results.items():
        c = colors[mode]
        smoothed = smooth(data["steps"])
        ax.plot(smoothed, color=c["avg"], linewidth=1.5, label=f"{mode}", alpha=0.8)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Steps")
    ax.set_title("Durée des épisodes (steps)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # ── 6. Loss Actor vs Critic ───────────────────────────────────────────
    ax = axes[2, 1]
    for mode, data in results.items():
        c = colors[mode]
        s_actor  = smooth(data["actor_losses"])
        s_critic = smooth(data["critic_losses"])
        ax.plot(s_actor, color=c["avg"], linewidth=1.5, linestyle='-',
                label=f"{mode} Actor", alpha=0.8)
        ax.plot(s_critic, color=c["avg"], linewidth=1.5, linestyle='--',
                label=f"{mode} Critic", alpha=0.5)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Loss")
    ax.set_title("Décomposition : Loss Actor (—) vs Critic (--)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison_modes.png", dpi=150)
    print("\n✅ Figure sauvegardée sous 'comparison_modes.png'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Dispositif : {DEVICE}")
    print(f"Jeu : {GAME}")
    print(f"Épisodes par mode : {N_EPISODES}")
    print(f"Deux entraînements séquentiels : LIFE puis EPISODE")

    # ── Infos GPU ─────────────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go")

    results = {}

    # ── Premier entraînement : mode LIFE ─────────────────────────────────
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results["LIFE"] = train_with_mode("LIFE")

    # ── Libérer la mémoire GPU entre les deux entraînements ──────────────
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        print(f"\n  💾 VRAM pic (LIFE) : {torch.cuda.max_memory_allocated()/1024**3:.2f} Go")
        torch.cuda.reset_peak_memory_stats()

    # ── Deuxième entraînement : mode EPISODE ─────────────────────────────
    results["EPISODE"] = train_with_mode("EPISODE")

    if DEVICE.type == "cuda":
        print(f"\n  💾 VRAM pic (EPISODE) : {torch.cuda.max_memory_allocated()/1024**3:.2f} Go")

    # ── Comparaison graphique ─────────────────────────────────────────────
    plot_comparison(results)

    # ── Résumé final ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*80}")
    print(f"  {'Métrique':<30s} {'LIFE':>12s} {'EPISODE':>12s}")
    print(f"  {'-'*54}")

    life = results["LIFE"]
    ep   = results["EPISODE"]

    print(f"  {'Moy finale (100 ép.)':<30s} {life['moving_avg'][-1]:>12.2f} {ep['moving_avg'][-1]:>12.2f}")
    print(f"  {'Meilleure moyenne':<30s} {life['best_avg']:>12.2f} {ep['best_avg']:>12.2f}")
    print(f"  {'Score max':<30s} {life['max_score']:>12.0f} {ep['max_score']:>12.0f}")
    print(f"  {'Steps moyen/épisode':<30s} {np.mean(life['steps']):>12.0f} {np.mean(ep['steps']):>12.0f}")
    print(f"  {'Updates moyen/épisode':<30s} {np.mean(life['updates']):>12.1f} {np.mean(ep['updates']):>12.1f}")
    print(f"  {'Seg. moyen (steps)':<30s} {np.mean(life['seg_lens']):>12.0f} {np.mean(ep['seg_lens']):>12.0f}")
    print(f"  {'Entropie finale':<30s} {life['entropies'][-1]:>12.3f} {ep['entropies'][-1]:>12.3f}")
    print(f"  {'Temps total':<30s} {life['total_time']/60:>11.1f}m {ep['total_time']/60:>11.1f}m")
    print(f"{'='*80}")
