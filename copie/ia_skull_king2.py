import skull_king_last_update2 as sk
import pygame
import random

nb_de_joueur = 4
input_dims = 60 + nb_de_joueur
# Ensure input_dims is correctly handled
input_dims = (input_dims,) if isinstance(input_dims, int) else input_dims

agent_parameters = {
    1: {
        'gamma': 0.95,
        'epsilon_jouer': 1.0,
        'epsilon_predire': 1.0,
        'lr': 0.001,
        'input_dims': input_dims,  # Exemple de dimension d'entrée
        'n_actions': 21, # nb carte max + nb prediction max 11 prédictions possible max
        'max_mem_size': 10000,
        'eps_end': 0.005,
        'eps_dec_jouer': 0.001,
        'eps_dec_predire': 0.0005
    },
    2: {
        'gamma': 0.9, # 0.8
        'epsilon_jouer': 1,
        'epsilon_predire': 0.6,
        'lr': 0.001, # pour adam
        'input_dims': input_dims,  # Exemple de dimension d'entrée
        'n_actions': 21, # nb carte max + nb prediction max 11 prédictions possible max
        'max_mem_size': 10000,
        'eps_end': 0.005,
        'eps_dec_jouer': 5e-4,
        'eps_dec_predire': 1e-3
    },
    3: {
        'gamma': 0.7, # 0.9
        'epsilon_jouer': 0.6,
        'epsilon_predire': 0.6,
        'lr': 0.001,
        'input_dims': input_dims,  # Exemple de dimension d'entrée
        'n_actions': 21, # nb carte max + nb prediction max 11 prédictions possible max
        'max_mem_size': 10000,
        'eps_end': 0.005,
        'eps_dec_jouer': 1e-4,
        'eps_dec_predire': 5e-4
    },
    4: {
        'gamma': 0.9, # 0.95
        'epsilon_jouer': 0.6,
        'epsilon_predire': 0.6,
        'lr': 0.001, # pour adam
        'input_dims': input_dims,  # Exemple de dimension d'entrée
        'n_actions': 21, # nb carte max + nb prediction max 11 prédictions possible max
        'max_mem_size': 10000,
        'eps_end': 0.005,
        'eps_dec_jouer': 1e-4,
        'eps_dec_predire': 5e-4
    },
}

# Définir la taille de la fenêtre
largeur_fenetre = 1000 # 1000 portable, 1200 fixe
hauteur_fenetre = 650 # 650 portable, 850 fixe
pygame.display.set_caption("Skull King")


def choix_nb_pli_aleatoire(joueur):
    return random.randrange(len(joueur.main+1)) # de 0 à nb_pli

def choix_pirate_drapeau_aleatoire():
    return random.choice(["D", "P"])

affichage_ecran = False

if affichage_ecran:
    fenetre = pygame.display.set_mode((largeur_fenetre, hauteur_fenetre))
else:
    fenetre = None

# Créer une instance de la classe Partie avec 3 joueurs
noms_joueurs = ["agent_1", "agent_2", "agent_3", "agent_4"]
partie = sk.Partie(noms_joueurs, fenetre=fenetre, affichage_ecran=affichage_ecran, choix_pirate_drapeau_aleatoire=choix_pirate_drapeau_aleatoire, agent_params=agent_parameters)

# Lancer la partie
partie.entrainement_ia()

# Quitter Pygame après la fin de la partie
pygame.quit()

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124