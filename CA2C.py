"""
TP5 – A2C Convolutionnel sur Atari (ALE/Breakout-v5)
======================================================
VERSION AMÉLIORÉE avec :
  1. Bonus d'entropie (exploration)
  2. Normalisation des avantages (stabilité)
  3. Reward clipping [-1, +1]
  4. Initialisation orthogonale des poids
  5. Fire-on-reset (Breakout nécessite l'action FIRE pour lancer la balle)
  6. Life-aware training (perte de vie = signal négatif)
  7. Learning rate scheduler (decay progressif)
  8. ★ Architecture IMPALA ResNet (extraction de features bien plus riche)

Mise à jour des paramètres : UNE SEULE FOIS à la fin de chaque épisode.
Architecture CNN : IMPALA ResNet (Espeholt et al., 2018) – bien plus profonde
que la Nature DQN, avec des blocs résiduels pour un meilleur flux de gradient.
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2

# ── Enregistrement des environnements Atari ──────────────────────────────────
gym.register_envs(ale_py)

# ── Hyperparamètres ───────────────────────────────────────────────────────────
GAME         = "ALE/Breakout-v5"   # Jeu choisi
K_FRAMES     = 4                   # Nombre de frames empilées
GAMMA        = 0.99                # Facteur d'actualisation
LR           = 7e-4                # Taux d'apprentissage Adam (augmenté)
CV           = 0.5                 # Coefficient perte critique
ENTROPY_COEF = 0.01                # ★ NOUVEAU : coefficient d'entropie pour l'exploration
GRAD_CLIP    = 0.5                 # Seuil gradient clipping (τ)
N_EPISODES   = 15000               # Entraînement long pour convergence complète
PRINT_EVERY  = 50                  # Affichage tous les N épisodes
SAVE_EVERY   = 1000                # ★ Sauvegarde checkpoint tous les N épisodes
UPDATE_MODE  = "EPISODE"           # Mode EPISODE uniquement (LIFE ne converge pas)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositif utilisé : {DEVICE}")
print(f"Jeu : {GAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRÉTRAITEMENT VISUEL
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    RGB (210,160,3) → Niveaux de gris → Redimensionné 84×84 → Normalisé [0,1].
    Retourne un tableau (84, 84) float32.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """
    Maintient un buffer circulaire de K frames prétraitées.
    L'état st ∈ R^(K×84×84) est obtenu en empilant K frames consécutives.

    Pourquoi empiler K images au lieu d'une seule ?
    → Une seule image est un "snapshot" statique. On ne peut PAS déduire
      la direction ni la vitesse de la balle/raquette. L'empilement de K=4
      frames permet au CNN de capturer le MOUVEMENT (différences inter-frames)
      et la VITESSE (amplitude des déplacements entre frames).
    """
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
# 2.  ARCHITECTURE IMPALA ResNet (Espeholt et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Bloc résiduel de l'architecture IMPALA :
      out = x + Conv(ReLU(Conv(ReLU(x))))

    Pourquoi les skip connections ?
    → Dans un réseau profond, le gradient doit traverser TOUTES les couches
      pour mettre à jour les premiers poids. Il peut devenir très petit
      (vanishing gradient). Le skip connection crée un "raccourci" direct :
      le gradient peut passer à travers sans atténuation.
      Résultat : on peut empiler beaucoup plus de couches convolutionnelles
      sans perdre la capacité d'apprentissage.
    """
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
        return out + residual  # ← skip connection


class ConvSequence(nn.Module):
    """
    Séquence convolutionnelle IMPALA :
      Conv2d(in→out) → MaxPool(3×3, stride=2) → ResBlock × 2

    Chaque ConvSequence :
      - Augmente le nombre de canaux (features de plus en plus abstraites)
      - Réduit la résolution spatiale de moitié (MaxPool)
      - Raffine les features via 2 blocs résiduels

    Comparaison avec Nature DQN :
      Nature DQN : 1 Conv par "étage" → 3 couches total → features basiques
      IMPALA     : 1 Conv + 2 ResBlocks par "étage" → 5 couches par étage
                   × 3 étages = 15 couches → features BEAUCOUP plus riches
    """
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


class A2CNet(nn.Module):
    """
    Réseau Actor-Critic avec backbone IMPALA ResNet :
      - Backbone CNN IMPALA : 3 ConvSequences (32→64→64 canaux)
        Chaque ConvSequence = Conv + MaxPool + 2 ResBlocks
        Total : 15 couches convolutionnelles (vs 3 pour Nature DQN)
      - Tête Acteur  : logits → πθ(a|st) = Softmax(gθ(zt))
      - Tête Critique: scalaire → Vϕ(st) = hϕ(zt)

    ★ Pourquoi IMPALA plutôt que Nature DQN ?
      → Le CNN Nature DQN (3 couches) extrait des features BASIQUES :
        bords, formes simples. C'est suffisant pour des jeux triviaux.
      → IMPALA ResNet (15 couches + skip connections) extrait des
        features HIÉRARCHIQUES : positions relatives, trajectoires,
        patterns de briques. Le réseau "comprend" mieux la scène.
      → Les skip connections permettent au gradient de traverser
        les 15 couches sans s'évanouir → convergence stable.
    """
    def __init__(self, n_actions: int, k_frames: int = K_FRAMES):
        super().__init__()

        # ── Backbone CNN IMPALA ResNet ──────────────────────────────────────
        #   3 étages de ConvSequence avec canaux croissants :
        #   (K_FRAMES, 84, 84) → (32, 42, 42) → (64, 21, 21) → (64, 11, 11)
        channels = [32, 64, 64]
        self.conv_sequences = nn.ModuleList()
        in_ch = k_frames
        for out_ch in channels:
            self.conv_sequences.append(ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        cnn_out_size = self._get_cnn_out(k_frames)

        # ── Couche FC après le CNN ──────────────────────────────────────────
        #   IMPALA utilise 256 neurones (vs 512 pour Nature DQN) car le CNN
        #   extrait déjà des features bien plus riches → moins besoin de
        #   capacité dans la couche FC.
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 256),
            nn.ReLU(),
        )

        # ── Tête Acteur ─────────────────────────────────────────────────────
        self.actor  = nn.Linear(256, n_actions)
        # ── Tête Critique ───────────────────────────────────────────────────
        self.critic = nn.Linear(256, 1)

        # ★ Initialisation orthogonale des poids
        self._init_weights()

    def _init_weights(self):
        """
        Initialisation adaptée à l'architecture IMPALA ResNet :

        1. Conv2d générales : orthogonale avec gain ReLU (√2)
        2. Conv2 dans les ResidualBlocks : ZÉRO (technique "Fixup")
           → Chaque ResidualBlock commence comme une IDENTITÉ :
             out = x + 0 = x
           Sans ça, le signal est amplifié à chaque bloc résiduel
           (×√2 par conv × 6 blocs = explosion des activations →
           logits énormes → softmax déterministe → entropie = 0)
        3. Tête Acteur : gain très faible (0.01)
           → Logits initiaux ≈ 0 → Softmax ≈ uniforme → Entropie maximale
           → L'agent EXPLORE avant de converger
        4. Tête Critique : gain très faible (0.01)
           → Valeurs initiales ≈ 0 (pas de biais initial)
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # ★ Fixup : zero-init la 2ème conv de chaque ResidualBlock
        #   Cela rend chaque bloc résiduel = identité au départ.
        #   Le réseau apprend PROGRESSIVEMENT à utiliser les résidus.
        for conv_seq in self.conv_sequences:
            nn.init.zeros_(conv_seq.res_block1.conv2.weight)
            nn.init.zeros_(conv_seq.res_block1.conv2.bias)
            nn.init.zeros_(conv_seq.res_block2.conv2.weight)
            nn.init.zeros_(conv_seq.res_block2.conv2.bias)

        # Tête acteur : gain faible → logits ≈ 0 → politique uniforme
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Tête critique : gain faible → valeurs initiales proches de 0
        nn.init.orthogonal_(self.critic.weight, gain=0.01)

    def _get_cnn_out(self, k_frames: int) -> int:
        """Calcule dynamiquement la taille de sortie du CNN IMPALA."""
        dummy = torch.zeros(1, k_frames, 84, 84)
        for conv_seq in self.conv_sequences:
            dummy = conv_seq(dummy)
        dummy = torch.relu(dummy)
        return int(dummy.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor):
        """
        x : (batch, K, 84, 84)  float32 dans [0,1]
        Retourne : logits (batch, n_actions), valeur (batch,)
        """
        # ── Passage à travers les 3 ConvSequences IMPALA ────────────────────
        for conv_seq in self.conv_sequences:
            x = conv_seq(x)

        # ── ReLU final + aplatissement ──────────────────────────────────────
        x = torch.relu(x)
        x = x.reshape(x.size(0), -1)

        # ── Couches FC + têtes Actor-Critic ─────────────────────────────────
        z      = self.fc(x)
        logits = self.actor(z)
        value  = self.critic(z).squeeze(-1)
        return logits, value


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALCUL DES RETOURS Rt (bootstrap = 0 si épisode terminé)
# ─────────────────────────────────────────────────────────────────────────────
def compute_returns(rewards: list, dones: list, gamma: float) -> list:
    """
    Rt = rt + γ·(1 - done_t)·R_{t+1}
    RT = 0 car l'épisode est terminé.
    """
    T       = len(rewards)
    returns = [0.0] * T
    g       = 0.0
    for t in reversed(range(T)):
        g          = rewards[t] + gamma * (1.0 - dones[t]) * g
        returns[t] = g
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MISE À JOUR DU MODÈLE (fonction utilitaire)
# ─────────────────────────────────────────────────────────────────────────────
MAX_BATCH_SIZE = 256  # ★ Taille max de mini-batch pour gradient accumulation
                      #   256 pour exploiter les 24 Go VRAM du L4


def update_model(model, optimizer, states, actions, rewards, dones):
    """
    Effectue UNE mise à jour A2C sur un segment de trajectoire.
    ★ GRADIENT ACCUMULATION : forward + backward par mini-batch pour
      borner la VRAM indépendamment de la longueur de l'épisode.
      Chaque mini-batch calcule sa perte partielle, fait backward()
      (les gradients s'accumulent), puis on fait UN SEUL optimizer.step().
    Retourne un dict avec les métriques détaillées, ou None si segment trop court.
    """
    if len(rewards) < 2:
        return None

    # ── Calcul des retours Rt ────────────────────────────────────────────
    returns_np = compute_returns(rewards, dones, GAMMA)
    returns_t  = torch.FloatTensor(returns_np).to(DEVICE)

    # ── Préparer les tenseurs ────────────────────────────────────────────
    states_np = np.array(states)
    actions_t = torch.LongTensor(actions).to(DEVICE)

    n = len(states)
    n_chunks = (n + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE  # nombre de mini-batches

    # ── Passe 1 : Forward sans gradient pour calculer les avantages ──────
    #   On a besoin des valeurs V(st) pour calculer les avantages AVANT
    #   de faire le vrai forward+backward. Cela ne consomme pas de VRAM
    #   car no_grad() ne stocke pas les activations intermédiaires.
    with torch.no_grad():
        values_list = []
        for i in range(0, n, MAX_BATCH_SIZE):
            chunk = torch.FloatTensor(states_np[i:i+MAX_BATCH_SIZE]).to(DEVICE)
            _, values_c = model(chunk)
            values_list.append(values_c)
        values_all = torch.cat(values_list, dim=0)

    # ── Avantage At = Rt − V(st) ────────────────────────────────────────
    advantages = returns_t - values_all

    # Normalisation des avantages → gradients plus stables
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Passe 2 : Gradient accumulation par mini-batch ───────────────────
    #   Chaque mini-batch fait forward → loss → backward().
    #   Les gradients s'ACCUMULENT dans model.parameters().grad.
    #   À la fin, on fait UN SEUL optimizer.step().
    optimizer.zero_grad()

    total_loss = 0.0
    total_actor = 0.0
    total_critic = 0.0
    total_entropy = 0.0

    for i in range(0, n, MAX_BATCH_SIZE):
        j = min(i + MAX_BATCH_SIZE, n)

        chunk_states  = torch.FloatTensor(states_np[i:j]).to(DEVICE)
        chunk_actions = actions_t[i:j]
        chunk_returns = returns_t[i:j]
        chunk_advs    = advantages[i:j]

        logits, values = model(chunk_states)

        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(chunk_actions)

        # Pertes sur ce mini-batch
        L_actor   = -(log_probs * chunk_advs.detach()).mean()
        L_critic  = ((chunk_returns - values) ** 2).mean()
        L_entropy = dist.entropy().mean()

        # On divise par n_chunks pour que la somme des gradients
        # = la moyenne globale (comme si on avait fait un seul gros batch)
        chunk_loss = (L_actor + CV * L_critic - ENTROPY_COEF * L_entropy) / n_chunks
        chunk_loss.backward()

        # Métriques (pour l'affichage seulement)
        total_loss    += chunk_loss.item() * n_chunks
        total_actor   += L_actor.item()
        total_critic  += L_critic.item()
        total_entropy += L_entropy.item()

    # ── UN SEUL step d'optimisation ──────────────────────────────────────
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        "loss": total_loss / n_chunks,
        "L_actor": total_actor / n_chunks,
        "L_critic": total_critic / n_chunks,
        "entropy": total_entropy / n_chunks,
        "seg_len": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def train():
    env       = gym.make(GAME)
    n_actions = env.action_space.n
    stacker   = FrameStack(K_FRAMES)

    model     = A2CNet(n_actions, K_FRAMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ★ Learning rate scheduler : décroissance linéaire plus douce
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,       # LR finale = LR × 0.2 (plus doux pour training long)
        total_iters=N_EPISODES
    )

    print(f"Actions disponibles : {n_actions}")
    print(f"Observations : {env.observation_space}")
    print(f"Mode de mise à jour : {UPDATE_MODE}")
    print(model)

    episode_rewards = []
    moving_avg      = []
    window          = deque(maxlen=100)
    best_avg        = 0.0

    for ep in range(1, N_EPISODES + 1):

        # ── Réinitialisation ─────────────────────────────────────────────────
        obs, info = env.reset()
        state     = stacker.reset(obs)

        # ★ Fire-on-reset : Breakout nécessite l'action FIRE (1) pour lancer
        #   la balle après chaque reset et après chaque perte de vie.
        obs, _, terminated, truncated, info = env.step(1)  # FIRE
        if not (terminated or truncated):
            state = stacker.step(obs)

        # ★ Life-aware : on surveille le nombre de vies pour détecter les pertes
        lives = info.get("lives", 5)

        # Buffers pour le segment courant (entre deux pertes de vie ou épisode)
        seg_states  = []
        seg_actions = []
        seg_rewards = []
        seg_dones   = []

        ep_reward     = 0.0
        done          = False
        last_entropy  = 0.0   # Dernière entropie mesurée (pour l'affichage)
        n_updates     = 0     # Nombre de mises à jour dans cet épisode

        # ── Collecte de la trajectoire ────────────────────────────────────────
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, _ = model(state_t)

            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ★ Life-aware : détecter la perte de vie
            new_lives = info.get("lives", lives)
            life_lost = (new_lives < lives)
            lives     = new_lives

            # ★ Reward clipping : borner les récompenses dans [-1, +1]
            clipped_reward = np.clip(reward, -1.0, 1.0)

            # ★ effective_done : coupe le retour Rt à chaque perte de vie
            effective_done = float(done or life_lost)

            next_state = stacker.step(next_obs)

            seg_states.append(state)
            seg_actions.append(action)
            seg_rewards.append(clipped_reward)
            seg_dones.append(effective_done)

            ep_reward += reward  # Récompense BRUTE pour le suivi
            state      = next_state

            # ── Mode LIFE : mise à jour à chaque perte de vie ────────────
            if UPDATE_MODE == "LIFE" and life_lost and not done:
                result = update_model(
                    model, optimizer,
                    seg_states, seg_actions, seg_rewards, seg_dones
                )
                if result is not None:
                    last_entropy = result["entropy"]
                    n_updates += 1
                # Réinitialiser les buffers du segment
                seg_states  = []
                seg_actions = []
                seg_rewards = []
                seg_dones   = []

            # ★ Si l'agent a perdu une vie, appuyer sur FIRE pour relancer
            if life_lost and not done:
                obs, _, terminated, truncated, info = env.step(1)
                if not (terminated or truncated):
                    state = stacker.step(obs)
                else:
                    done = True

        # ── Mise à jour sur le segment restant (fin d'épisode) ────────────────
        #   Mode EPISODE : c'est la seule update (toute la trajectoire)
        #   Mode LIFE    : update sur le dernier segment (après dernière vie)
        if len(seg_rewards) >= 2:
            result = update_model(
                model, optimizer,
                seg_states, seg_actions, seg_rewards, seg_dones
            )
            if result is not None:
                last_entropy = result["entropy"]
                n_updates += 1

        scheduler.step()  # ★ Décroissance du learning rate (1 step par épisode)

        # ── Suivi des performances ───────────────────────────────────────────
        episode_rewards.append(ep_reward)
        window.append(ep_reward)
        avg = np.mean(window)
        moving_avg.append(avg)

        if avg > best_avg:
            best_avg = avg
            torch.save(model.state_dict(), "a2c_breakout_best.pth")

        # ★ Checkpoint périodique (pour reprendre si interruption)
        if ep % SAVE_EVERY == 0:
            checkpoint = {
                "episode": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_avg": best_avg,
                "episode_rewards": episode_rewards,
                "moving_avg": moving_avg,
            }
            torch.save(checkpoint, f"checkpoint_ep{ep}.pth")
            print(f"  💾 Checkpoint sauvegardé : checkpoint_ep{ep}.pth")

        if ep % PRINT_EVERY == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Épisode {ep:5d}/{N_EPISODES} | "
                  f"Récomp : {ep_reward:5.1f} | "
                  f"Moy(100) : {avg:6.2f} | "
                  f"Best : {best_avg:6.2f} | "
                  f"Entropie : {last_entropy:.3f} | "
                  f"Updates : {n_updates} | "
                  f"LR : {current_lr:.2e}")

    env.close()
    return episode_rewards, moving_avg, model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(episode_rewards: list, moving_avg: list):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(episode_rewards, alpha=0.3, color="steelblue",
            label="Récompense par épisode")
    ax.plot(moving_avg, color="darkorange", linewidth=2,
            label="Moyenne glissante (100 épisodes)")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense totale")
    ax.set_title(f"A2C Convolutionnel – {GAME.split('/')[-1]} (amélioré)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("a2c_rewards.png", dpi=150)
    print("Figure sauvegardée sous 'a2c_rewards.png'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    episode_rewards, moving_avg, model = train()
    plot_results(episode_rewards, moving_avg)

    torch.save(model.state_dict(), "a2c_breakout_final.pth")
    print("Modèle final sauvegardé sous 'a2c_breakout_final.pth'")
    print("Meilleur modèle sauvegardé sous 'a2c_breakout_best.pth'")
