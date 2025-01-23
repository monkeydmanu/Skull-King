import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

"""
def choix_indice_aleatoire_parmi_indice_carte_dispo(indice_carte_dispo):
    return random.choice(indice_carte_dispo)

def choix_indice_aleatoire_parmi_nb_pli_max(nb_pli_max_possible):
    return random.randrange(nb_pli_max_possible+1)
"""
def choix_indice_aleatoire(n_actions):
    return random.randrange(n_actions) # de 0 à 21 non inclus donc 21 actions possibles

# Création du modèl
class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, hidden_units, n_actions, lr, device):
        super().__init__()
        self.input_shape = input_shape[0] if isinstance(input_shape, tuple) else input_shape # changer d'approche si on passe des tuples pour de vrai
        self.hidden_units = hidden_units[0] if isinstance(hidden_units, tuple) else hidden_units
        self.n_actions = n_actions[0] if isinstance(n_actions, tuple) else n_actions
        self.device = device

        self.block1 = nn.Sequential(
            nn.Linear(in_features=self.input_shape, out_features=self.hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_units, out_features=self.n_actions)
        ).to(self.device)
        """
            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
        """
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return (self.block1(x))
    #nn.Softmax(dim=-1)
# utiliser SOFTMAX a la place de sigmoid


class BaseAgent:
    def __init__(self, gamma, epsilon_jouer, epsilon_predire, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size=10000, eps_end=0.05, eps_dec_jouer=5e-4, eps_dec_predire=5e-3):
        self.illegal_moves_count = 0
        self.illegal_moves_per_game = []
        self.loss_learn = []
        self.loss_learn_seul = []
        # Suivi des valeurs d'epsilon
        self.epsilon_jouer_history = [epsilon_jouer]
        self.epsilon_predire_history = [epsilon_predire]
        self.gamma = gamma
        self.epsilon_jouer = epsilon_jouer  # Epsilon pour jouer
        self.epsilon_predire = epsilon_predire  # Epsilon pour prédire les plis
        self.eps_min = eps_end
        self.eps_dec_jouer = eps_dec_jouer
        self.eps_dec_predire = eps_dec_predire
        self.lr = lr
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.input_dims = input_dims

        # Modèle pour jouer des cartes
        self.Q_eval_jouer = DeepQNetwork(input_shape=input_dims, hidden_units=256, n_actions=n_actions_jouer, lr=self.lr, device=device)

        # Modèle pour prédire les plis
        self.Q_eval_predire = DeepQNetwork(input_shape=input_dims, hidden_units=256, n_actions=n_actions_predire, lr=self.lr, device=device)


        # Mémoires pour les transitions
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.plis_memory = np.zeros(self.mem_size, dtype=np.int32)  # Mémoire pour les prédictions de plis


    def reset(self):
        self.illegal_moves_per_game.append(self.illegal_moves_count)
        self.illegal_moves_count = 0
        # Réinitialiser les compteurs de mémoire et d'itérations
        self.mem_cntr = 0
        self.iter_cntr = 0
        # Réinitialiser les mémoires
        self.state_memory.fill(0)
        self.new_state_memory.fill(0)
        self.action_memory.fill(0)

    def update_epsilon(self):
        # Suivi des valeurs d'epsilon
        self.epsilon_jouer_history.append(self.epsilon_jouer)
        self.epsilon_predire_history.append(self.epsilon_predire)

    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.Q_eval.state_dict(),
            'optimizer_state_dict': self.Q_eval.optimizer.state_dict(),
            'epsilon_jouer': self.epsilon_jouer,
            'mem_cntr': self.mem_cntr,
            'iter_cntr': self.iter_cntr,
            'epsilon_predire': self.epsilon_predire
        }
        T.save(checkpoint, filename)
        print(f"Agent saved successfully to {filename}.")

    def load(self, filename):
        checkpoint = T.load(filename)
        self.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon_jouer = checkpoint['epsilon_jouer']
        self.mem_cntr = checkpoint['mem_cntr']
        self.iter_cntr = checkpoint['iter_cntr']
        self.epsilon_predire = checkpoint['epsilon_predire']
        print(f"Agent loaded successfully from {filename}.")

    def clean_illegal_transitions(self):
        self.illegal_moves_count += 1
        # print(f"\n\n illegal transition : {self.mem_cntr = }")
        if self.mem_cntr > 0:
            # Calculer l'index du dernier élément
            index = (self.mem_cntr - 1) % self.mem_size

            # Déplacer les éléments suivants vers la gauche pour combler le vide
            self.state_memory[index] = np.zeros_like(self.state_memory[index])
            self.new_state_memory[index] = np.zeros_like(self.new_state_memory[index])
            self.action_memory[index] = 0

            # Réduire le compteur de mémoire
            self.mem_cntr -= 1

    def store_transition(self, state, action, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = np.array(state, dtype=np.float32)
        self.new_state_memory[index] = np.array(state_, dtype=np.float32)
        self.action_memory[index] = action
        self.mem_cntr += 1


class Agent(BaseAgent):
    def __init__(self, gamma, epsilon_jouer, epsilon_predire, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size=10000, eps_end=0.05, eps_dec_jouer=5e-4, eps_dec_predire=5e-3):
        super().__init__(gamma, epsilon_jouer, epsilon_predire, lr, input_dims, n_actions_jouer, n_actions_predire, max_mem_size, eps_end, eps_dec_jouer, eps_dec_predire)

    def choose_action(self, observation, is_predire=None):
        if not is_predire:  # Jouer une carte
            epsilon = self.epsilon_jouer
            if np.random.random() > epsilon:
                state = T.tensor(np.array(observation), dtype=T.float32).to(self.Q_eval_jouer.device)
                actions = self.Q_eval_jouer.forward(state)
                action = T.argmax(actions).item()
            else:
                action = choix_indice_aleatoire(self.n_actions_jouer)
        else:  # Prédire un pli
            epsilon = self.epsilon_predire
            if np.random.random() > epsilon:
                state = T.tensor(np.array(observation), dtype=T.float32).to(self.Q_eval_predire.device)
                actions = self.Q_eval_predire.forward(state)
                action = T.argmax(actions).item()
            else:
                action = choix_indice_aleatoire(self.n_actions_predire)
        
        return action
    
    # dans learn j'appprends avec la meilleurs des 10 valeurs qui m'intéressent
    def learn_tout(self, reward):
        
        print("LEARN TOUT")
        print(f"{reward = }")

        self.Q_eval.optimizer.zero_grad()

        suite_indice_state_selectionner = list(range(55))

        state_batch = T.tensor(self.state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval.device)

        action_batch = self.action_memory[suite_indice_state_selectionner]

        q_eval = self.Q_eval.forward(state_batch) # [:, :10]
        q_next= self.Q_eval.forward(new_state_batch) # [:, :10]

        # Calcul de la perte pour la sélection des actions
        q_eval_actions = q_eval[suite_indice_state_selectionner, action_batch] # prend les actions prises dans les derniers states
        q_target = reward + self.gamma * T.max(q_next, dim=1)[0]

        print(f"{suite_indice_state_selectionner = }")
        print(f"{q_eval_actions = }\n{q_target = }\n{reward = }")
        loss_action = self.Q_eval.loss(q_eval_actions, q_target).to(self.Q_eval.device)
        # print(f"{loss_action = }")
        self.loss_learn.append(loss_action.item())
        """
        if reward < 0:
            print(f"C'est de la merde {loss_action = }")
        else:
            print(f"C'est niquel {loss_action = }")
            if loss_action.item() > 100:
                print(f"{q_eval_actions = }\n{q_target = }")
        """
        loss_action.backward()
        self.Q_eval.optimizer.step()
        print(f"{q_eval[suite_indice_state_selectionner, action_batch] = }")

        self.iter_cntr += 1
        self.epsilon_jouer = self.epsilon_jouer - (self.eps_dec_jouer*55) if self.epsilon_jouer > self.eps_min else self.eps_min # pour décroître par unité d'état

        # print(f"---------------------------------------\n, {self.epsilon_jouer = }")
        # print(f"---------------------------------------\n, {self.epsilon_predire = }")
        # print("---------------------------------------")

        # print(f"{suite_indice_state_selectionner = }\n {batch_index = }\n {action_batch = }\n {state_batch = }\n {new_state_batch = }\n {action_batch = }\n \
            #    {q_eval = }\n {q_next = }\n {q_eval_actions = }\n {q_target = }\n {loss_action = }\n {reward = }")
        
        # Mise à jour de l'epsilon après apprentissage
        self.update_epsilon()


    # dans learn j'appprends avec la meilleurs des 10 valeurs qui m'intéressent
    def learn(self, reward):

        print("LEARN")
        print(f"{reward = }")
        # Calculer la fonction de perte combinée pour la sélection des actions et la prédiction des plis
        # Implémentation similaire à votre fonction learn, mais avec deux sorties à gérer
        if self.mem_cntr > 0:
            indice = (self.mem_cntr - 1) % self.mem_size
        else :
            indice = 0

        nb_manche = self.state_memory[indice][self.input_dims[0] -4]  # On récupère le numéro de la manche en cours

        self.Q_eval.optimizer.zero_grad()

        if self.mem_cntr > 0:
            suite_indice_state_selectionner = list(range((self.mem_cntr) - int(nb_manche), self.mem_cntr))
        else:
            suite_indice_state_selectionner = list(range(self.mem_cntr))
        batch_index = np.arange(nb_manche, dtype=np.int32) # le nombre de pli jouer

        state_batch = T.tensor(self.state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[suite_indice_state_selectionner], dtype=T.float32).to(self.Q_eval.device)

        action_batch = self.action_memory[suite_indice_state_selectionner]

        q_eval = self.Q_eval.forward(state_batch) # [:, :10]
        q_next= self.Q_eval.forward(new_state_batch) # [:, :10]

        # Calcul de la perte pour la sélection des actions
        q_eval_actions = q_eval[batch_index, action_batch] # prend les actions prises dans les derniers states
        q_target = reward + self.gamma * T.max(q_next, dim=1)[0] #[0] car T.max retourne à la fois le max et l'indice du max

        print(f"{suite_indice_state_selectionner = }")
        print(f"{q_eval[batch_index] = }")
        print(f"{q_eval_actions = }\n{q_target = }\n{reward = }")
        loss_action = self.Q_eval.loss(q_eval_actions, q_target).to(self.Q_eval.device)
        # print(f"{loss_action = }")
        self.loss_learn.append(loss_action.item())
        """
        if reward < 0:
            print(f"C'est de la merde {loss_action = }")
        else:
            print(f"C'est niquel {loss_action = }")
            if loss_action.item() > 100:
                print(f"{q_eval_actions = }\n{q_target = }")
        """
        loss_action.backward()
        self.Q_eval.optimizer.step()

        print(f"{q_eval[batch_index, action_batch] = }")

        self.iter_cntr += 1
        self.epsilon_jouer = self.epsilon_jouer - (self.eps_dec_jouer*nb_manche) if self.epsilon_jouer > self.eps_min else self.eps_min # pour décroître par unité d'état

        # print(f"---------------------------------------\n, {self.epsilon_jouer = }")
        # print(f"---------------------------------------\n, {self.epsilon_predire = }")
        # print("---------------------------------------")

        # print(f"{suite_indice_state_selectionner = }\n {batch_index = }\n {action_batch = }\n {state_batch = }\n {new_state_batch = }\n {action_batch = }\n \
            #    {q_eval = }\n {q_next = }\n {q_eval_actions = }\n {q_target = }\n {loss_action = }\n {reward = }")
        
        # Mise à jour de l'epsilon après apprentissage
        self.update_epsilon()

    # dans learn_seul j'appprends avec la meilleurs des 20 valeurs
    def learn_seul(self, reward, plis_gagnes=None, is_predire=False):

        print("LEARN SEUL")
        print(f"{reward = }")
        # print(f"{reward = }")
        if self.mem_cntr > 0:
            indice = (self.mem_cntr - 1) % self.mem_size
        else :
            indice = 0


        # pour le learn_seul à la fin de la manche
        # on prend l'état sur lequel on a prédit pour le comparé avec le nb de plis gagnes
        nb_manche = self.state_memory[indice][self.input_dims[0] -4]
        if self.mem_cntr > 0:
            indice_state_prediction_debut = self.mem_cntr - int(nb_manche)
        else:
            indice_state_prediction_debut = 0

        self.Q_eval.optimizer.zero_grad()

        # old_weights = [param.clone() for param in self.Q_eval.parameters()]

        # pour le learn_seul d'une mauvaise prédiction
        if is_predire and plis_gagnes is None:
            print("mauvaise prédiction")
            state_batch = T.tensor(self.state_memory[indice]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[indice]).to(self.Q_eval.device)
            action_batch = self.action_memory[indice]
            q_eval_tmp = self.Q_eval.forward(state_batch)
            print(f"{q_eval_tmp = }")
            q_eval = q_eval_tmp[action_batch]
            q_next = self.Q_eval.forward(new_state_batch)

            """
            # Appliquer le masque pour les 11 dernières valeurs sur les 21
            mask = T.zeros_like(q_eval_tmp, dtype=T.bool)
            mask[10:] = True  # Active seulement les 11 dernières

            # Sélectionner uniquement les valeurs masquées pour le calcul de la perte
            q_next_masked = T.masked_select(q_next, mask)
            """

            # Calcul du Q-target avec le masquage
            q_target = reward + self.gamma * T.max(q_next)

        # pour learn_seul d'une mauvais action
        elif plis_gagnes is None:
            print("mauvais action")
            state_batch = T.tensor(self.state_memory[indice]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[indice]).to(self.Q_eval.device)
            action_batch = self.action_memory[indice]
            # print(f"{self.Q_eval.forward(state_batch) = }")
            q_eval_tmp = self.Q_eval.forward(state_batch)
            q_eval = q_eval_tmp[action_batch]
            q_next = self.Q_eval.forward(new_state_batch)

            """
            # Masque pour garder uniquement les 10 premières valeurs
            mask = T.zeros_like(q_eval_tmp, dtype=T.bool)
            mask[:10] = True  # Active seulement les 10 premières valeurs sur les 21

            # Sélectionner les valeurs masquées pour la perte
            q_next_masked = T.masked_select(q_next, mask)
            """
            # Calcul du Q-target avec les valeurs masquées
            q_target = reward + self.gamma * T.max(q_next)

        # pour le learn_seul de la prédiction à la fin
        else:
            print("prédiction à la fin")
            state_batch = T.tensor(self.state_memory[indice_state_prediction_debut]).to(self.Q_eval.device)
            action_batch = self.action_memory[indice_state_prediction_debut]
            # print(f"{self.Q_eval.forward(state_batch) = }")
            q_eval_tmp = self.Q_eval.forward(state_batch)
            q_eval = q_eval_tmp[action_batch]

            q_target = T.zeros_like(q_eval)  # Crée un tensor avec la même forme que q_eval
            q_target = reward + self.gamma * T.tensor(1, dtype=T.float32).to(self.Q_eval.device)  # Assigner la valeur 1 pour représenter le nombre de plis réellement gagnés
            # print("--------------------------------------------------------------------")
            # print(f"{indice_state_prediction_debut = }")

        # Calcul de la perte
        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        self.loss_learn_seul.append(loss.item())
        # print(f"{q_eval = }\n{q_target = }\n{reward = }")
        # print(f"{loss = }")
        """
        if reward < 0:
            print(f"C'est de la merde {loss = }")
        else:
            print(f"C'est niquel {loss = }")
        """
        loss.backward()

        """
        with T.no_grad():
            params = list(self.Q_eval.parameters())
            last_layer_index = len(params) - 1  # Index de la dernière couche
            for idx, param in enumerate(params):
                if param.grad is not None:
                    if idx == last_layer_index:
                        if is_predire and plis_gagnes is None:
                            # Dernière couche : appliquer un masque basé sur les sorties
                            if param.dim() == 2:  # Matrice de poids
                                # Par exemple, pour ne pas modifier les 10 premières sorties
                                param.grad[:10, :] = 0
                            elif param.dim() == 1:  # Biais
                                param.grad[:10] = 0
                        elif plis_gagnes is None:
                            if param.dim() == 2:  # Matrice de poids
                                # Par exemple, pour ne pas modifier les 10 premières sorties
                                param.grad[10:, :] = 0
                            elif param.dim() == 1:  # Biais
                                param.grad[10:] = 0
                        else:
                            if param.dim() == 2:  # Matrice de poids
                                # Par exemple, pour ne pas modifier les 10 premières sorties
                                param.grad[:10, :] = 0
                            elif param.dim() == 1:  # Biais
                                param.grad[:10] = 0
        """


        self.Q_eval.optimizer.step()

        """
        # Vérification des poids après mise à jour
        with T.no_grad():
            for idx, param in enumerate(self.Q_eval.parameters()):
                if param.grad is not None:
                    # Calcul des indices modifiés
                    changed = (param - old_weights[idx]).abs() > 1e-6
                    
                    # Vérification selon le contexte de la modification
                    if is_predire and plis_gagnes is None:
                        # Vérifier que seuls les indices des 11 dernières valeurs ont changé
                        unexpected_changes = changed[:10]
                    elif plis_gagnes is None:
                        # Vérifier que seuls les indices des 10 premières valeurs ont changé
                        unexpected_changes = changed[10:]
                    else:
                        # Vérifier que seuls les indices des 11 dernières valeurs ont changé
                        unexpected_changes = changed[:10]
                    
                    
                    # Affichage si des indices ignorés ont changé
                    if unexpected_changes.any():
                        print(f"Attention : Les indices ignorés ont été modifiés pour le paramètre {idx}.")
        """

        
        print(f"{self.Q_eval.forward(state_batch) = }")

        if is_predire:
            self.epsilon_predire = max(self.epsilon_predire - self.eps_dec_predire, self.eps_min)
        else:
            self.epsilon_jouer = max(self.epsilon_jouer - self.eps_dec_jouer, self.eps_min)

        # Mise à jour de l'epsilon après apprentissage
        self.update_epsilon()

    # Visualisation des graphiques
    def plot_all_graphs(self, joueur):
        # Initialisation de la figure avec 2 lignes et 2 colonnes
        plt.figure(figsize=(14, 10))

        # 1er graphique : Quantité de coups illégaux par jeu
        plt.subplot(2, 2, 1)
        plt.plot(self.illegal_moves_per_game, marker='o', color='r')
        plt.title(f'Quantité de coups illégaux par jeu pour l\'{joueur.nom}')
        plt.xlabel('Jeu')
        plt.ylabel('Coups illégaux')

        # 2ème graphique : Perte pour learn
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_learn, label='Loss Learn', color='b')
        plt.title(f'Perte en fonction du temps pour l\'{joueur.nom}')
        plt.xlabel('Itérations')
        plt.ylabel('Perte')
        plt.legend()

        # 3ème graphique : Perte pour learn_seul
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_learn_seul, label='Loss Learn Seul', color='g')
        plt.title(f'Perte learn_seul en fonction du temps pour l\'{joueur.nom}')
        plt.xlabel('Itérations')
        plt.ylabel('Perte')
        plt.legend()

        # 4ème graphique : Évolution des epsilon_jouer et epsilon_predire
        plt.subplot(2, 2, 4)
        plt.plot(self.epsilon_jouer_history, label='Epsilon Jouer', color='b')
        plt.plot(self.epsilon_predire_history, label='Epsilon Prédire', color='g')
        plt.title('Évolution de Epsilon en fonction du temps')
        plt.xlabel('Itérations')
        plt.ylabel('Valeur d\'Epsilon')
        plt.legend()
        plt.grid(True)

        # Affichage de tous les graphiques
        plt.tight_layout()
        plt.show()
