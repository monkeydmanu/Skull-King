import pygame
import random
import time
import model
import os
import torch as T
import inspect
import logging

def attendre(duree_ms):
    """
    Fonction qui simule une pause non bloquante pendant 'duree_ms' millisecondes.
    """
    start_time = pygame.time.get_ticks()  # Temps de départ
    while pygame.time.get_ticks() - start_time < duree_ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        pygame.display.flip()  # Actualise l'affichage pour éviter le blocage

# Configuration du premier logger pour écrire dans partie2.log
logger_partie = logging.getLogger('logger_partie2')
logger_partie.setLevel(logging.INFO)
# Handler pour écrire dans le fichier partie2.log
handler_partie = logging.FileHandler('partie3joueurs.log', mode='w')
handler_partie.setFormatter(logging.Formatter('%(message)s'))
logger_partie.addHandler(handler_partie)
"""
# Handler pour afficher dans le terminal
console_handler_partie = logging.StreamHandler()
console_handler_partie.setFormatter(logging.Formatter('%(message)s'))
logger_partie.addHandler(console_handler_partie)
"""

"""
# Configuration du deuxième logger pour écrire dans test1.log
logger_test = logging.getLogger('logger_test1')
logger_test.setLevel(logging.INFO)
# Handler pour écrire dans le fichier test1.log
handler_test = logging.FileHandler('test1.log', mode='w')  # Ouvre en mode 'w' pour écraser à chaque exécution
handler_test.setFormatter(logging.Formatter('%(message)s'))
logger_test.addHandler(handler_test)
"""
"""
# Handler pour afficher dans le terminal
console_handler_test = logging.StreamHandler()
console_handler_test.setFormatter(logging.Formatter('%(message)s'))
logger_test.addHandler(console_handler_test)
"""


# Initialiser Pygame
pygame.init()

# Définir la taille de la fenêtre
largeur_fenetre = 1000 # 1000 portable, 1200 fixe
hauteur_fenetre = 650 # 650 portable, 850 fixe

# Définir la taille des cartes redimensionnées
taille_carte = (100, 150)  # Largeur et hauteur des cartes

x_start = 200  # Position X de départ pour les cartes du joueur
y_start = hauteur_fenetre - taille_carte[1] - 20  # Position Y de départ pour les cartes du joueur

x_start_pli = 200
y_start_pli = hauteur_fenetre - 400


espace_carte = 80  # Espacement entre les cartes

# Charger les images des cartes
def charger_images():
    images = {}
    couleurs = ["jaune", "bleu", "rouge", "noire"]
    valeurs = list(range(1, 15))
    for couleur in couleurs:
        for valeur in valeurs:
            image = pygame.image.load(f"images/{valeur}_{couleur}.PNG")
            images[f"{valeur}_{couleur}"] = pygame.transform.scale(image, taille_carte)
    images["Pirate"] = [pygame.transform.scale(pygame.image.load(f"images/pirate_{i}.PNG"), taille_carte) for i in range(1, 6)]
    images["Skull King"] = pygame.transform.scale(pygame.image.load("images/skull_king.PNG"), taille_carte)
    images["Sirène"] = pygame.transform.scale(pygame.image.load("images/sirene.PNG"), taille_carte)
    images["Tigresse"] = pygame.transform.scale(pygame.image.load("images/tigresse.PNG"), taille_carte)
    images["Drapeau"] = pygame.transform.scale(pygame.image.load("images/drapeau.PNG"), taille_carte)
    images["tigresse_pirate"] = pygame.transform.scale(pygame.image.load("images/tigresse_pirate.PNG"), taille_carte)
    images["tigresse_drapeau"] = pygame.transform.scale(pygame.image.load("images/tigresse_drapeau.PNG"), taille_carte)
    images["fond"] = pygame.transform.scale(pygame.image.load("images/fond_pirate.png"), (largeur_fenetre, hauteur_fenetre))
    return images

# Fonction pour afficher une carte
def afficher_carte(fenetre, images, carte, x, y, choix):
    image = None
    
    if carte.valeur in range(1, 15) and carte.couleur in ["jaune", "bleu", "rouge", "noire"]:
        image = images.get(f"{carte.valeur}_{carte.couleur}", None)
    elif carte.valeur == "Pirate":
        image = images["Pirate"][0]  # Utiliser la première image pour Pirate
    elif carte.valeur == "Skull King":
        image = images["Skull King"]
    elif carte.valeur == "Sirène":
        image = images["Sirène"]
    elif carte.valeur == "Tigresse" and choix == "D":
        image = images["tigresse_drapeau"]
    elif carte.valeur == "Tigresse" and choix == "P":
        image = images["tigresse_pirate"]
    elif carte.valeur == "Tigresse":
        image = images["Tigresse"]
    elif carte.valeur == "Drapeau":
        image = images["Drapeau"]

    if image:
        fenetre.blit(image, (x, y))

def afficher_cartes_pli(fenetre, images, pli, x, y, choix):
    for index, (joueur, carte) in enumerate(pli):
        afficher_carte(fenetre, images, carte, x + index * (espace_carte), y, choix)

def afficher_info_joueur(fenetre, joueur):
    x = 20
    y = hauteur_fenetre - 150
    font = pygame.font.Font(None, 30)

    # fond noir des infos affichées
    pygame.draw.rect(fenetre, (0, 0, 0), (x-5, y-5, 170, 140))


    # Texte à afficher (nom du joueur, prédiction, plis gagnés, points, points bonus)
    texte_lignes = [
        joueur.nom,
        "",
        f"Prédiction : {joueur.prediction}",
        f"Plis gagnés : {joueur.plis_gagnes}",
        f"Points : {joueur.points}",
        f"Points bonus : {joueur.points_bonus}"
    ]
    
    # Couleur du texte (blanc)
    couleur_texte = (255, 255, 255)
    
    # Afficher chaque ligne de texte séparément
    for i, ligne in enumerate(texte_lignes):
        # Rendu de chaque ligne de texte
        texte_rendu = font.render(ligne, True, couleur_texte)
        
        # Calcul de la position y pour chaque ligne
        y_ligne = y + i * (font.get_linesize())
        
        # Affichage de la ligne de texte à la position donnée
        fenetre.blit(texte_rendu, (x, y_ligne))


# Fonction pour afficher les cartes du joueur
def afficher_cartes_joueur(fenetre, images, joueur, x, y):
    for index, carte in enumerate(joueur.main):
        afficher_carte(fenetre, images, carte, x + index * (espace_carte), y, None)

# Fonction pour afficher les choix "Pirate" et "Drapeau" pour la carte Tigresse
def afficher_choix_tigresse(fenetre):
    font = pygame.font.Font(None, 36)
    pygame.draw.rect(fenetre, (255, 255, 255), (largeur_fenetre // 2 - 100, hauteur_fenetre // 2 - 50, 90, 100))
    texte_pirate = font.render("Pirate", True, (0, 0, 0))
    fenetre.blit(texte_pirate, (largeur_fenetre // 2 - 90, hauteur_fenetre // 2 - 20))
    pygame.draw.rect(fenetre, (255, 255, 255), (largeur_fenetre // 2 + 20, hauteur_fenetre // 2 - 50, 110, 100))
    texte_drapeau = font.render("Drapeau", True, (0, 0, 0))
    fenetre.blit(texte_drapeau, (largeur_fenetre // 2 + 30, hauteur_fenetre // 2 - 20))

# Fonction pour détecter la carte cliquée
def detecter_carte_au_clique(x, y, joueur):
    
    for index, carte in enumerate(joueur.main):
        rect_x = x_start + index * (espace_carte)
        rect_y = y_start
        if len(joueur.main) == 1:
            if rect_x <= x <= rect_x + taille_carte[0] and rect_y <= y <= rect_y + taille_carte[1]:
                return index
        else:
            if rect_x <= x <= rect_x + espace_carte and rect_y <= y <= rect_y + taille_carte[1]: # en effet les cartes ont une taille réduites pour l'affichage
                return index
    return None

# Classe Carte (inchangée)
class Carte:
    def __init__(self, valeur, couleur):
        self.valeur = valeur
        self.couleur = couleur

    def __repr__(self):
        return f"{self.valeur} {self.couleur}"

    def __eq__(self, other):
        if isinstance(other, Carte):
            return self.valeur == other.valeur and self.couleur == other.couleur
        return False

    def __hash__(self):
        return hash((self.valeur, self.couleur))

# Classe JeuDeCartes (inchangée)
class JeuDeCartes:
    couleurs = ["jaune", "bleu", "rouge", "noire"]
    valeurs = list(range(1, 15))

    def __init__(self):
        self.initialiser_jeu()

    def initialiser_jeu(self):
        self.cartes = [Carte(valeur, couleur) for couleur in self.couleurs for valeur in self.valeurs]
        self.cartes += [Carte("Pirate", "spécial")] * 5
        self.cartes += [Carte("Skull King", "spécial"), Carte("Sirène", "spécial"), Carte("Tigresse", "spécial")]
        self.cartes += [Carte("Drapeau", "spécial")] * 5

    def melanger(self):
        random.shuffle(self.cartes)

    def distribuer(self, nb_cartes):
        main = [self.cartes.pop() for _ in range(nb_cartes)]
        return main

# Classe Joueur (inchangée)
class Joueur:
    def __init__(self, nom, nb_de_joueur, is_agent=False, agent_params=None, numero_agent=None):
        self.nb_de_joueur = nb_de_joueur
        self.agent_params = agent_params
        self.nom = nom
        self.main = []
        self.prediction = 0
        self.plis_gagnes = 0
        self.points = 0
        self.points_bonus = 0
        self.is_agent = is_agent
        self.numero_agent = numero_agent
        self.agent = self.init_agent() if is_agent else None

    def init(self):
        self.main = []
        self.prediction = 0
        self.plis_gagnes = 0
        self.points = 0
        self.points_bonus = 0

    def init_agent(self):
        agent = model.Agent(self.agent_params[self.numero_agent]['gamma'],
                                 self.agent_params[self.numero_agent]['epsilon_jouer'],
                                 self.agent_params[self.numero_agent]['epsilon_predire'],
                                 self.agent_params[self.numero_agent]['lr'],
                                 self.agent_params[self.numero_agent]['input_dims'],
                                 self.agent_params[self.numero_agent]['n_actions_jouer'],
                                 self.agent_params[self.numero_agent]['n_actions_predire'],
                                 self.agent_params[self.numero_agent]['max_mem_size'],
                                 self.agent_params[self.numero_agent]['eps_end'],
                                 self.agent_params[self.numero_agent]['eps_dec_jouer'],
                                 self.agent_params[self.numero_agent]['eps_dec_predire']
                                 )
        # Charger les poids du modèle s'ils existent
        model_path = f"models{self.nb_de_joueur}/agent_jouer{int(self.nom[-1])}.pth"
        if os.path.exists(model_path):
            agent.load(model_path, agent.Q_eval_jouer)
            print(f"Modèle chargé pour {int(self.nom[-1])} depuis {model_path}")
        else:
            print(f"Aucun modèle trouvé pour jouer {int(self.nom[-1])}, initialisation avec des poids par défaut.")

        model_path = f"models{self.nb_de_joueur}/agent_predire{int(self.nom[-1])}.pth"
        if os.path.exists(model_path):  
            agent.load(model_path, agent.Q_eval_predire)
            print(f"Modèle chargé pour {int(self.nom[-1])} depuis {model_path}")
        else:
            print(f"Aucun modèle trouvé pour predire {int(self.nom[-1])}, initialisation avec des poids par défaut.")
        return agent


    def jouer_carte(self, index):
        return self.main.pop(index)


    def __repr__(self):
        return f"{self.nom} :\n Points: {self.points},\n Points bonus: {self.points_bonus} \n {self.main = }"

        
# Classe Partie avec ajout de l'affichage Pygame
class Partie:
    def __init__(self, joueurs, fenetre, affichage_ecran=True,
                 choix_indice_aleatoire_parmi_indice_carte_dispo=None, 
                 choix_nb_pli_aleatoire=None, choix_pirate_drapeau_aleatoire=None,
                  agent_params=None):
        self.fenetre = fenetre
        self.affichage_ecran = affichage_ecran
        self.agent_params = agent_params if agent_params else None
        self.joueurs = self.init_joueur(joueurs)
        self.jeu_de_cartes = JeuDeCartes()
        self.tour = 1
        self.premier_joueur_index = 0
        self.images = charger_images() if self.affichage_ecran else None# Charger les images des cartes
        self.phase_prediction = True  # Phase de prédiction des plis
        self.carte_tigresse = False  # Détection de la carte Tigresse
        self.choix_tigresse = None  # Choix pour la carte Tigresse
        self.plis = []
        self.choix_indice_aleatoire_parmi_indice_carte_dispo = choix_indice_aleatoire_parmi_indice_carte_dispo
        self.choix_nb_pli_aleatoire = choix_nb_pli_aleatoire
        self.choix_pirate_drapeau_aleatoire = choix_pirate_drapeau_aleatoire

    def initialiser(self):
        self.jeu_de_cartes = JeuDeCartes()
        self.tour = 1
        self.phase_prediction = True  # Phase de prédiction des plis
        self.carte_tigresse = False  # Détection de la carte Tigresse
        self.choix_tigresse = None  # Choix pour la carte Tigresse
        self.plis = []


    def init_joueur(self, nom_joueurs):
        joueurs = []
        for nom in nom_joueurs:
            if nom.startswith("agent"):
                joueurs.append(Joueur(nom, len(nom_joueurs), is_agent=True, agent_params=self.agent_params, numero_agent=int(nom[-1])))
            else:
                joueurs.append(Joueur(nom, len(nom_joueurs)))
        return joueurs

    def reordonner_joueurs(self, gagnant, joueurs_tour):
        index_gagnant = joueurs_tour.index(gagnant)
        joueurs_tour = joueurs_tour[index_gagnant:] + joueurs_tour[:index_gagnant]
        return joueurs_tour

    def jouer_manche(self):
        print(f"Tour {self.tour} :")
        self.jeu_de_cartes.initialiser_jeu()
        self.jeu_de_cartes.melanger()
        nb_cartes = self.tour
        for joueur in self.joueurs:
            joueur.main = self.jeu_de_cartes.distribuer(nb_cartes)
        joueurs_tour = self.joueurs[self.premier_joueur_index:] + self.joueurs[:self.premier_joueur_index]

        if self.phase_prediction:
            for joueur in joueurs_tour:
                self.predire_plis(joueur)
            self.phase_prediction = False

        for tour in range(self.tour):
            # Jouer le pli
            joueurs_tour = self.jouer_pli(joueurs_tour)
        


        # Passer à la phase suivante (recalculer les points)
        self.calculer_points()
        self.tour += 1

        self.phase_prediction = True
        
        # Mettre à jour le premier joueur pour le prochain tour
        self.premier_joueur_index = (self.premier_joueur_index + 1) % len(self.joueurs)


    def reinitialiser_points(self):
        for joueur in self.joueurs:
            joueur.plis_gagnes = 0
            joueur.points_bonus = 0

    def predire_plis(self, joueur):
        self.plis = []
        choix_prediction = None
        if self.choix_nb_pli_aleatoire:
            if self.afficher_ecran:
                self.afficher_ecran(joueur, None)
            choix = self.choix_nb_pli_aleatoire(joueur)
            joueur.prediction = choix
            # print(f"{joueur.nom} a prédit aleatoirement {choix} plis.")
            if self.affichage_ecran:
                attendre(1000)
        elif joueur.is_agent:
            while choix_prediction is None:
                # print(f"\n{joueur.nom}")
                # Obtenir l'état actuel
                # print(f"Actuel {joueur = }")
                state = self.obtenir_etat_actuel(joueur, None)

                choix_prediction = joueur.agent.choose_action(state, None, nb_prediction_max=self.tour)
                # print(f"{choix_prediction+10 = }")

                # logger_test.info(f"\n{joueur.main = }\n{self.tour = }")
                # logger_test.info(f"{choix_prediction=}")
                # Vérification de la validité de l'action
                if choix_prediction > self.tour or choix_prediction < 0:
                    # print(f"ta fais de la D pour predire, tu as choisi {choix_prediction+10}")
                    # Si l'action est illégale, on apprend directement un malus et réinitialise l'état et rejouer
                    # print(f"{joueur.nom}")
                    reward = -1  # Pénalité pour coup illégal
                    state_ = state  # Pas de changement d'état car l'action est annulée
                    joueur.agent.store_transition_predire(state, choix_prediction, state_) # +10 pour prédiction

                    # print("mauvaise prediction")
                    joueur.agent.learn_seul(reward, is_predire=True) # cas sans connaître plis gagnes
                    # Nettoyage des transitions illégales à la fin de la manche
                    joueur.agent.clean_illegal_transitions_predire()
                    choix_prediction = None  # Réinitialiser pour refaire jouer l'agent
                else:
                    # print(f"{joueur.nom} a prédit {choix_prediction} plis.")
                    joueur.prediction = choix_prediction
                    reward = 0.1
                    indice_carte_dispo = self.carte_valide_a_jouer(joueur, None) # couleur départ à None
                    state_ = self.obtenir_etat_actuel(joueur, indice_carte_dispo)
                    joueur.agent.store_transition_predire(state, choix_prediction, state_)
                
                    joueur.agent.learn_seul(reward, is_predire=True)

        else:
            choix = None
            while choix is None:
                self.afficher_ecran(joueur, None) # evidemmene que l'écran doit être affiché
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        choix = self.detecter_prediction(x, y)
                        if choix is not None:
                            # print(f"{joueur.nom} a prédit {choix} plis.")
                            joueur.prediction = choix

    # Créer le vecteur d'observation
    """
    obs = [v1, ca1, v2, ca2, v3, ca3, v4, ca4, v5, ca5, v6, ca6, v7, ca7, v8, ca8, v9, ca9, v10, ca10,# valeur carte dans la main du joueur
                # caracteristique carte dans la main du joueur
       b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, #indice_carte_possible_de_jouer,
       nombre_cartes_dans_la_main,
       nb_joueurs,
       phase_prediction
        prediction, plis_gagnes,
        nb_carte_pli, v1, ca1, v2, ca2, v3, ca3, v4, ca4, v5, ca5, v6, ca6, v7, ca7, v8, ca8, v9, ca9, v10, ca10, # cartes dans le pli
        numero_manche, point, point_bonus,
        choix_tigresse]
    """
    def obtenir_etat_actuel(self, joueur, indice_carte_dispo):
        dictionnaire_caracteristiques = {
            "jaune": 0,
            "bleu": 1,
            "rouge": 2,
            "noire": 3,
            "spécial": 4
        }

        # Dictionnaire pour convertir les cartes spéciales en valeurs numériques > 14
        dictionnaire_cartes_speciales = {
            "Drapeau": 0.5,
            "Sirène": 16,
            "Pirate": 17,
            "Tigresse": 18,
            "Skull King": 19
        }

        state = []

        # 1. Carte dans la main du joueur (valeur et caractéristique)
        cartes_joueur = []
        for carte in joueur.main:
            # Vérifier si la valeur est un entier sinon utiliser le dictionnaire des cartes spéciales
            valeur = dictionnaire_cartes_speciales.get(carte.valeur, carte.valeur)
            couleur = [0,0,0,0,0]
            indice = dictionnaire_caracteristiques[carte.couleur]
            couleur[indice] = 1
            cartes_joueur.append(valeur)
            cartes_joueur.extend(couleur)

        # Compléter avec des zéros si moins de 10 cartes (chaque carte a maintenant 2 positions)
        cartes_joueur += [-1, -1, -1, -1, -1, -1] * (10 - len(joueur.main)) # un zéro pour la valeur puis 5 pour la couleur 
        state.extend(cartes_joueur)

        # 2. Indices des cartes possibles à jouer
        if indice_carte_dispo:
            indices_cartes_dispo = [1 if i in indice_carte_dispo else 0 for i in range(10)] # booléen avec 1 et 2, voir si ça marche
        else:
            indices_cartes_dispo = [0 for i in range(10)]

        state.extend(indices_cartes_dispo)

        # 3. Nombre de cartes dans la main non normalisé
        nombre_cartes_dans_la_main = len(joueur.main)
        state.append(nombre_cartes_dans_la_main)

        # 4. Nombre de joueurs
        nb_joueurs = len(self.joueurs)
        state.append(nb_joueurs)

        # 4 Bis phase prédiction ou pas
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)
        phase_prediction = self.phase_prediction
        state.append(phase_prediction)

        # 5. Prédiction et plis gagnés du joueur
        state.append(joueur.prediction)
        state.append(joueur.plis_gagnes)

        # 5.2 Prédiction de chaque joueur
        for joueur_du_jeu in self.joueurs:
            if not self.phase_prediction: # si ce n'est pas la phase de prédiction
                state.append(joueur_du_jeu.prediction)
            else:
                state.append(0)
        joueurs_absents = [-1] * (10 - len(self.joueurs))
        state.extend(joueurs_absents)

        # 6. Nombre de cartes dans le pli
        nb_carte_pli = len(self.plis)
        state.append(nb_carte_pli)

        # 7. Cartes dans le pli (valeur et caractéristique)
        cartes_pli = []
        for joueur_pli, carte in self.plis:
            valeur = dictionnaire_cartes_speciales.get(carte.valeur, carte.valeur)
            couleur = [0,0,0,0,0]
            indice = dictionnaire_caracteristiques[carte.couleur]
            couleur[indice] = 1
            cartes_pli.append(valeur)
            cartes_pli.extend(couleur)

        # Compléter avec des zéros si moins de 10 cartes
        cartes_pli += [-1, -1, -1, -1, -1, -1] * (10 - len(self.plis))
        state.extend(cartes_pli)

        # 8. Numéro de la manche
        state.append(self.tour)

        # 9. Points et points bonus du joueur
        state.append(joueur.points)
        state.append(joueur.points_bonus)

        return state


    def calculer_points_bonus(self, gagnant):
        """
        calcule les points bonus et les attributs

        Arguments :
            gagnant (Joueur) : Le joueur gagnant du pli.
        """
        for _, carte in self.plis:  # Parcourt les cartes du pli
            if carte.valeur == 14:  # Si une carte de valeur 14 est trouvée
                gagnant.points_bonus += 1
                if carte.couleur == "noire":
                    gagnant.points_bonus += 1

        # Vérifier si un pirate ou une tigresse en mode pirate du gagnant est dans le pli
        pirate_du_gagnant = any(
            (carte.valeur == "Pirate" or 
            (carte.valeur == "Tigresse" and self.choix_tigresse == "P")) and 
            joueur == gagnant
            for joueur, carte in self.plis
        )

        if pirate_du_gagnant:
            # Compter le nombre de sirènes dans le pli
            sirenes = sum(
                1 for _, carte in self.plis if carte.valeur == "Sirène"
            )
            gagnant.points_bonus += 2 * sirenes

        # Si le gagnant est le Skull King, compter les pirates et tigresses
        skull_king_gagnant = any(
            carte.valeur == "Skull King" and joueur == gagnant
            for joueur, carte in self.plis
        )

        if skull_king_gagnant:
            # Compter les pirates et tigresses (peu importe leur mode)
            pirates_tigresses = sum(
                1 for _, carte in self.plis 
                if carte.valeur == "Pirate" or carte.valeur == "Tigresse"
            )
            gagnant.points_bonus += 3 * pirates_tigresses

        # Si le gagnant est une Sirène, vérifier la présence du Skull King
        sirene_gagnante = any(
            carte.valeur == "Sirène" and joueur == gagnant
            for joueur, carte in self.plis
        )

        if sirene_gagnante:
            # Vérifier si le Skull King est dans le pli
            skull_king_present = any(
                carte.valeur == "Skull King"
                for _, carte in self.plis
            )
            if skull_king_present:
                gagnant.points_bonus += 4

    
    def jouer_pli(self, joueurs_tour):
        self.plis = []
        self.choix_tigresse = None
        couleur_depart = None
        for joueur in joueurs_tour:
            if self.affichage_ecran:
                self.afficher_ecran(joueur, self.choix_tigresse)
            carte_jouee = None
            indice_carte_dispo = self.carte_valide_a_jouer(joueur, couleur_depart)

            while carte_jouee is None:
                index = None
                if self.choix_indice_aleatoire_parmi_indice_carte_dispo: # juste pour une version complètement aléatoire
                            index = self.choix_indice_aleatoire_parmi_indice_carte_dispo(indice_carte_dispo)
                            # print(f"joueur carte aleatoire : {joueur.main[index] = }")
                            if self.affichage_ecran:
                                attendre(1000)
                elif joueur.is_agent:

                    # Obtenir l'état actuel
                    state = self.obtenir_etat_actuel(joueur, indice_carte_dispo)

                    index = joueur.agent.choose_action(state, indice_carte_dispo=indice_carte_dispo)
                    # print(f"{index = }")
                    # logger_test.info(f"\n{joueur.main = }\n{indice_carte_dispo = }")
                    # logger_test.info(f"{index=}")

                    # Vérification de la validité de l'action
                    if (index >= len(joueur.main)) or (index not in indice_carte_dispo):
                        # print(f"{joueur.nom}")
                        # print(f"ta fais de la D pour jouer, tu as choisi {index}")
                        # Si l'action est illégale, on apprend directement un malus et réinitialise l'état et rejouer
                        reward = -1  # Pénalité pour coup illégal, il ne faut pas que ce soit élevé car ont le fait souvent au début donc ça va mettre le 0 en avant au départ
                        state_ = state  # Pas de changement d'état car l'action est annulée

                        # print(f"{joueur = }")
                        joueur.agent.store_transition(state, index, state_)
                        joueur.agent.learn_seul(reward)
                        # Nettoyage des transitions illégales à la fin de la manche
                        joueur.agent.clean_illegal_transitions()
                        index = None  # Réinitialiser pour refaire jouer l'agent
                else :
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            index = detecter_carte_au_clique(x, y, joueur)
                if index is not None:
                    carte = joueur.main[index]
                    # l'ia va apprendre à gérer tout seul les coup valide avec les rewards
                    if index in indice_carte_dispo:
                        carte_jouee = joueur.jouer_carte(index)
                        # pour déterminer la couleur de départ
                        if couleur_depart is None and carte_jouee.couleur in JeuDeCartes.couleurs:
                            possible = True
                            for joueur_pli, carte in self.plis:
                                if carte.couleur == "spécial" and carte.valeur != "Drapeau" and self.choix_tigresse != "D":
                                    possible = False
                            if possible == True:
                                couleur_depart = carte_jouee.couleur
                        if carte_jouee == Carte("Tigresse", "spécial"):
                            if joueur.is_agent:
                                if self.choix_pirate_drapeau_aleatoire:
                                    self.choix_tigresse = self.choix_pirate_drapeau_aleatoire()
                                    # print(f"choix tigresse aleatoire : {self.choix_tigresse = }")
                                    if self.affichage_ecran:
                                        attendre(1000)
                            else:
                                self.choix_tigresse = self.jouer_tigresse(joueur)
                            self.carte_tigresse = False
                        self.plis.append((joueur, carte_jouee))
                        if joueur.is_agent:
                            if self.affichage_ecran:
                                attendre(1000)

                        if joueur.is_agent:
                            state_ = self.obtenir_etat_actuel(joueur, indice_carte_dispo)

                            reward = 0.1
                            joueur.agent.store_transition(state, index, state_)
                            # print(f"{joueur.nom}")
                            joueur.agent.learn_seul(reward)
                        if self.affichage_ecran:
                            self.afficher_ecran(joueur, self.choix_tigresse)
                            if self.affichage_ecran:
                                attendre(1000)
                    else:
                        print("pas possible")
                        carte = None
                        carte_jouee = None

                    

        # Déterminer le gagnant du pli
        gagnant = self.determiner_gagnant(couleur_depart)
        if gagnant:
            gagnant.plis_gagnes += 1
            self.calculer_points_bonus(gagnant)
        joueurs_tour = self.reordonner_joueurs(gagnant, joueurs_tour)
        return joueurs_tour


    def detecter_prediction(self, x, y):
        for i in range(self.tour + 1):
            rect_x = largeur_fenetre // 2 - 150 + (i - 1) * 60
            rect_y = hauteur_fenetre // 2 - 30
            if rect_x <= x <= rect_x + 50 and rect_y <= y <= rect_y + 50:
                return i
        return None

    def carte_valide_a_jouer(self, joueur, couleur_pli_depart):
        indice_carte_dispo = []
        for indice, carte in enumerate(joueur.main):
            if self.coup_valide(joueur, carte, couleur_pli_depart):
                indice_carte_dispo.append(indice)
        return indice_carte_dispo

    def coup_valide(self, joueur, carte_jouee, couleur):
        carte_gagnante = self.meilleure_carte(couleur)
        if couleur not in JeuDeCartes.couleurs:
            return True
        if carte_jouee.couleur == "spécial":
            return True
        if couleur == "noire":
            if carte_jouee.couleur == "noire":
                if carte_gagnante and carte_gagnante.couleur == "noire":
                    if any(carte.valeur > carte_gagnante.valeur for carte in joueur.main if carte.couleur == "noire"):
                        if carte_jouee.valeur < carte_gagnante.valeur:
                            return False
                    else:
                        return True
        if carte_jouee.couleur == couleur:
            return True
        if not any(carte.couleur == couleur for carte in joueur.main):
            return True
        return False

    def meilleure_carte(self, couleur_depart):
        if not self.plis:
            return None

        special_cards_priority = {
            "Skull King": 100,
            "Pirate": 99,
            "Sirène": 98,
            "Drapeau": 0
        }

        def carte_priority(carte, couleur_depart):
            if carte.valeur in special_cards_priority:
                return special_cards_priority[carte.valeur]

            if carte.couleur == "noire":
                return carte.valeur + 50

            if carte.couleur == couleur_depart:
                return carte.valeur + 30

            if carte.valeur == "Tigresse":
                if self.choix_tigresse == "D":
                    return 0
                elif self.choix_tigresse == "P":
                    return 99

            return carte.valeur

        plis_sorted = sorted(self.plis, key=lambda x: carte_priority(x[1], couleur_depart), reverse=True)
        return plis_sorted[0][1]

    def determiner_gagnant(self, couleur_depart):
        if not self.plis:
            return None

        meilleure_carte = self.meilleure_carte(couleur_depart)
        for joueur, carte in self.plis:
            if carte == meilleure_carte:
                return joueur

    def calculer_points(self):
        for joueur in self.joueurs:
            if joueur.prediction == joueur.plis_gagnes:
                joueur.points += self.tour + joueur.points_bonus
            elif joueur.prediction == 0:
                joueur.points -= self.tour
            else:
                joueur.points += 0

    def calculer_recompense_ia(self, joueur):
        if joueur.prediction == joueur.plis_gagnes :
            reward = 1
            if joueur.points_bonus > 0:
                reward += 1
        if joueur.prediction != joueur.plis_gagnes :
            reward = -1
            if joueur.prediction == 0:
                reward -= 2
        
        return reward

    def jouer_tigresse(self, joueur):
        self.carte_tigresse = True
        while self.carte_tigresse:
            if self.afficher_ecran:
                self.afficher_ecran(joueur, None)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if largeur_fenetre // 2 - 100 <= x <= largeur_fenetre // 2 and hauteur_fenetre // 2 - 50 <= y <= hauteur_fenetre // 2 + 50:
                        return "P"
                    elif largeur_fenetre // 2 + 20 <= x <= largeur_fenetre // 2 + 120 and hauteur_fenetre // 2 - 50 <= y <= hauteur_fenetre // 2 + 50:
                        return "D"
            pygame.display.flip()

    def detecter_carre_blanc(self, x, y, joueur):
        for index in range(len(joueur.main) + 1):
            rect_x = largeur_fenetre // 2 - 150 + (index - 1) * 60
            rect_y = hauteur_fenetre // 2 - 30
            if rect_x <= x <= rect_x + 50 and rect_y <= y <= rect_y + 50:
                return index
        return None

    def afficher_ecran(self, joueur, choix):
        """
        # Obtenir le cadre d'appel (call frame) du contexte actuel
        cadre_actuel = inspect.currentframe()
        # Accéder au cadre de l'appelant
        cadre_appelant = cadre_actuel.f_back
        # Obtenir le nom de la fonction appelante
        nom_fonction_appelante = cadre_appelant.f_code.co_name
        print(f"La fonction appelante est : {nom_fonction_appelante}")
        """
        self.fenetre.blit(self.images["fond"], (0, 0))
        afficher_info_joueur(self.fenetre, joueur)
        if self.plis:
            afficher_cartes_pli(self.fenetre, self.images, self.plis, x_start_pli, y_start_pli, choix)
        afficher_cartes_joueur(self.fenetre, self.images, joueur, x_start, y_start)
        if self.phase_prediction:
            self.afficher_predictions()
        else:
            if self.carte_tigresse:
                afficher_choix_tigresse(self.fenetre)
        pygame.display.flip()

    def afficher_predictions(self):
        font = pygame.font.Font(None, 36)
        for i in range(self.tour + 1):
            pygame.draw.rect(self.fenetre, (255, 255, 255), (largeur_fenetre // 2 - 150 + (i - 1) * 60, hauteur_fenetre // 2 - 30, 50, 50))
            text = font.render(str(i), True, (0, 0, 0))
            self.fenetre.blit(text, (largeur_fenetre // 2 - 140 + (i - 1) * 60, hauteur_fenetre // 2 - 20))

    def jouer(self):
        self.initialiser()
        for joueur in self.joueurs:
            joueur.init()
            if joueur.is_agent:
                joueur.agent.reset()
        # print(len(self.joueurs))
        # print(self.joueurs)
        # print(self.choix_indice_aleatoire_parmi_indice_carte_dispo, self.choix_nb_pli_aleatoire, self.choix_pirate_drapeau_aleatoire)
        self.premier_joueur_index = random.randint(0, len(self.joueurs) - 1)
        while self.tour <= 10:
            self.jouer_manche()
            for joueur in self.joueurs:
                if joueur.is_agent:
                    reward = self.calculer_recompense_ia(joueur)
                    # print(f"\n\n{joueur = }, \n\n{reward = }\n")
                    # print(f"\n{joueur.nom} - Prediction: {joueur.prediction}, Plis gagnes: {joueur.plis_gagnes}")
                    # print(f"{joueur.nom}")

                    joueur.agent.learn(reward)

                    joueur.agent.learn_seul(reward, plis_gagnes=joueur.plis_gagnes, is_predire=True)
            self.reinitialiser_points()
        
        for joueur in self.joueurs:
            print(f"{joueur.nom} - Points: {joueur.points}, Points bonus: {joueur.points_bonus}")
            if joueur.is_agent:
                print(f"{joueur.agent.epsilon_jouer = } \n {joueur.agent.epsilon_predire = }")
                joueur.agent.update_value_loss()
                joueur.agent.reset_game_loss()
        
        gagnant = max(self.joueurs, key=lambda j: j.points)

        for joueur in self.joueurs:
            if joueur.is_agent:
                if joueur.points < 0:
                    reward = -1 # test à -100 en cours
                    joueur.agent.learn_tout(reward)
        
        if gagnant.points > 0:
            if gagnant.is_agent:
                gagnant.agent.learn_tout(1)

        

        logger_partie.info("\n\n")
        for joueur in self.joueurs:
            logger_partie.info(f"{joueur.nom} - Points: {joueur.points}, Points bonus: {joueur.points_bonus}")
            if joueur.is_agent:
                logger_partie.info(f"epsilon_jouer: {joueur.agent.epsilon_jouer}, epsilon_predire: {joueur.agent.epsilon_predire}")
        logger_partie.info(f"\nLe gagnant est {gagnant.nom} avec {gagnant.points} points!")

    def entrainement_ia(self, nb_de_joueur):
        # n_agents = 4
        n_episodes = 50
        model_dir = f'models{nb_de_joueur}/'

        # Assurez-vous que le répertoire de modèles existe
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Boucle de jeu
        for episode in range(n_episodes):
            print(f"Episode {episode + 1}/{n_episodes}")
            self.jouer()
            if episode % 5 == 0: # % 25
                print("target_dqn synchronisé")
                for joueur in self.joueurs:
                    if joueur.is_agent:
                        joueur.agent.target_dqn_jouer.load_state_dict(joueur.agent.Q_eval_jouer.state_dict())
                        joueur.agent.target_dqn_predire.load_state_dict(joueur.agent.Q_eval_predire.state_dict())
            if episode % 50 == 0: 
                for joueur in self.joueurs:
                    if joueur.is_agent:
                        model_path = os.path.join(model_dir, f'agent_jouer{int(joueur.nom[-1])}.pth')
                        joueur.agent.save(model_path, joueur.agent.Q_eval_jouer)
                        model_path = os.path.join(model_dir, f'agent_predire{int(joueur.nom[-1])}.pth')
                        joueur.agent.save(model_path, joueur.agent.Q_eval_predire)


        # il faudrait qu'on puisse cliquer sur un bouton pour charger un modèle même si ça tourne encore pour pouvoir l'arrêter
        # Sauvegarde des modèles
        for joueur in self.joueurs:
            if joueur.is_agent:
                model_path = os.path.join(model_dir, f'agent_jouer{int(joueur.nom[-1])}.pth')
                joueur.agent.save(model_path, joueur.agent.Q_eval_jouer)
                print(f"Modèle de l'agent {int(joueur.nom[-1])} sauvegardé dans {model_path}")
                model_path = os.path.join(model_dir, f'agent_predire{int(joueur.nom[-1])}.pth')
                joueur.agent.save(model_path, joueur.agent.Q_eval_predire)
                print(f"Modèle de l'agent {int(joueur.nom[-1])} sauvegardé dans {model_path}")
                joueur.agent.plot_all_graphs(joueur)

        # il faut afficher les infos des bots au cours du temps avec torchinfo pourquoi pas
        # ainsi que les rewards

if __name__ == "__main__":
    noms_joueurs = ["Alice", "Bob", "Charlie"]
    partie = Partie(noms_joueurs)
    partie.jouer()




# print(f"\n\n {self.plis=}")