# Melting point approximation

Un test d'entrainement d'un petit model de machine learning pour approximer le point de fusion d'une molécule.

Il prend en entrée un vecteur embedded de 512 valeurs sorti d'un modèle de LSTM (https://github.com/jrwnter/cddd) prenant en entrée une représentation SMILES d'une molécule.

Le modèle est trés simple, et transforme l'entrée de 512 en une seule sortie via des activations linéaire.
On observe une différence moyenne entre 10 et 20 degrés pour l'ensemble du set.
