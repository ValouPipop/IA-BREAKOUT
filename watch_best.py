"""
Script pour VOIR le meilleur modèle A2C jouer à Breakout.
Charge a2c_breakout_best.pth et affiche le jeu en temps réel.
"""

import gymnasium as gym
import ale_py
import torch
import numpy as np
import cv2
from CA2C import A2CNet, FrameStack, K_FRAMES, DEVICE

# ── Enregistrement des environnements Atari ──────────────────────────────────
gym.register_envs(ale_py)

GAME = "ALE/Breakout-v5"
MODEL_PATH = "a2c_best_episode.pth"  # ← Ton meilleur modèle
N_GAMES = 5  # Nombre de parties à regarder


def watch_agent(render_mode="human"):
    """Charge le meilleur modèle et le fait jouer visuellement."""

    # ── Créer l'environnement avec rendu visuel ──────────────────────────────
    env = gym.make(GAME, render_mode=render_mode)
    n_actions = env.action_space.n
    stacker = FrameStack(K_FRAMES)

    # ── Charger le modèle ────────────────────────────────────────────────────
    model = A2CNet(n_actions, K_FRAMES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()  # Mode évaluation (désactive dropout, batchnorm, etc.)

    print(f"✅ Modèle chargé depuis '{MODEL_PATH}'")
    print(f"🎮 Dispositif : {DEVICE}")
    print(f"🕹️  Lancement de {N_GAMES} partie(s)...\n")

    total_rewards = []

    for game in range(1, N_GAMES + 1):
        obs, info = env.reset()
        state = stacker.reset(obs)

        # Fire pour lancer la balle
        obs, _, terminated, truncated, info = env.step(1)
        if not (terminated or truncated):
            state = stacker.step(obs)

        lives = info.get("lives", 5)
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, value = model(state_t)

            # Prendre l'action la plus probable (greedy) pour évaluation
            action = logits.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Détecter perte de vie et relancer
            new_lives = info.get("lives", lives)
            if new_lives < lives and not done:
                obs, _, terminated, truncated, info = env.step(1)  # FIRE
                if terminated or truncated:
                    done = True

            lives = new_lives
            state = stacker.step(obs)
            ep_reward += reward
            steps += 1

        total_rewards.append(ep_reward)
        print(f"  Partie {game}/{N_GAMES} | Score : {ep_reward:.0f} | Steps : {steps}")

    env.close()

    print(f"\n{'='*50}")
    print(f"📊 Résultats sur {N_GAMES} parties :")
    print(f"   Moyenne  : {np.mean(total_rewards):.1f}")
    print(f"   Max      : {np.max(total_rewards):.0f}")
    print(f"   Min      : {np.min(total_rewards):.0f}")
    print(f"   Écart-type : {np.std(total_rewards):.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    watch_agent()
