#!/usr/bin/env python3.10

import numpy as np
import matplotlib.pyplot as plt


class Modem :
    """
        Classe permettant d'implémenter un MODulateur/dEModulateur PAM ou ASK (2,4,8), QPSK et 16QAM.
    """

    def __init__(self, ModType, NbSymboles, bits):
        """
        Contructeur de la classe

        Parametres :
            ModType : type de modulation, PAM, ASK, PSK ou QAM
            NbSymboles : nombre de symboles de la modulation. 2, 4 ou 8 pour PAM ou ASK, 4 pour PSK et 16 pour QAM 
            bits : tableau de bits numpy       
        """
        self.modtype = ModType
        self.nsymb = NbSymboles
        self.mod = (ModType, NbSymboles)
        if ModType == 'PAM' or ModType == 'ASK' :
            self.symb_type = 'reel'
        else : 
            self.symb_type = 'complexe'
        self.bits = bits
        self.bits_par_symb = int(np.log2(self.nsymb))
        self.symbs_num = bits.reshape(int(len(bits)/self.bits_par_symb), self.bits_par_symb)
        if (self.nsymb & (self.nsymb-1) == 0) and self.nsymb != 0 :
            self.bit_par_symb=int(np.log2(self.nsymb))
        else :
            raise ValueError('La deuxième valeur qui correspond au nombre de symboles \
            doit être une puissance de 2 : 2, 4, 8, 16, 32, 64, ...')
            
    def create_MP(self, amplitude):
        """
        Fonction en charge de créer la table de mapping de chaque modulation
    
        Parametres
        ----------
        amplitude : amplitude maximale des sybmole de modulaiton pour une modulation PAM ou ASK,
                    amplitude max de la sinusoide pour une modulation PSK et amplitude max de I
                    et Q pour une modulation QAM
        phase_ori : utilisé seulement pour la modulation QPSK, phase à l'origine du premier
                    symbole (par déffaut = 0
   
        Retourne
        -------
        mapping_table : la table de mapping sous forme d'un dictionnaire
        """
        match self.mod:
            case ('PAM',2)| ('ASK',2) :
                mapping_table = {(0,) : -1,
                                (1,) : 1}
            case ('PAM',4)| ('ASK',4) :
                mapping_table = {(0,0) : -3,
                                (0,1) : -1,
                                (1,0) : 1,
                                (1,1) : 3}
            case ('PAM',8)| ('ASK',8) :
                mapping_table = {(0,0,0) : -7,
                                (0,0,1) : -5,
                                (0,1,0) : -3,
                                (0,1,1) : -1,
                                (1,0,0) : 1,
                                (1,0,1) : 3,
                                (1,1,0) : 5,
                                (1,1,1) : 7}   
            case _:
                mapping_table = None
                print(f'La modulation {self.nsymb}{self.modtype} n\'est pas implémentée')
        for key in mapping_table.keys() :
            mapping_table[key] = mapping_table[key] * amplitude / (self.nsymb-1)
        self.mapping_table = mapping_table
        return(mapping_table)

    def mapping(self, amplitude):
        self.mapping_table = self.create_MP(amplitude)
        symbs_mod=np.array([self.mapping_table[tuple(symb)] for symb in self.symbs_num])
        return(symbs_mod)
    
    
class Mesure:
    def __init__(self, signal): # Vérifie que le signal est un tableau numpy
        self.signal = signal

    def DSP(self, fe, type='Bi', unit='dBm'):  # Ajout de self comme premier argument
        signal = self.signal  # Utilisation du signal de l'instance
        N = signal.shape[0]  # Nombre de points du signal
        te = 1 / fe  # Période d'échantillonnage
        f_tranche = np.fft.fftfreq(N, d=te)  # Fréquences associées aux bins FFT

        # Calcul de la FFT bilatérale
        Y_fft = np.fft.fft(signal) / N  # Normalisation de la FFT

        if type == 'Bi':  # Affichage bilatéral
            Y_tranche_mod = np.abs(Y_fft)
            f_tranche = np.fft.fftshift(f_tranche)  # Décale la fréquence pour centrer sur 0 Hz
            Y_tranche_mod = np.fft.fftshift(Y_tranche_mod)  # Applique le même décalage sur le spectre
        elif type == 'mono':  # Affichage mono-latéral
            Y_tranche_mod = np.abs(Y_fft[:N // 2]) * 2  # Mono-latéral : on garde la moitié positive et double l'amplitude
            f_tranche = f_tranche[:N // 2]  # On garde la moitié des fréquences positives
        else:
            raise ValueError("Le paramètre 'type' doit être 'Bi' ou 'mono'.")

        if unit == 'Volts':  # Si l'unité demandée est en Volts efficaces
            Y_output = Y_tranche_mod / np.sqrt(2)  # En Volts efficaces
            ylabel = 'Amplitude (Volts efficaces)'
        elif unit == 'dBm':  # Sinon, on considère l'unité en dBm par défaut
            Y_output = 10 * np.log10(np.square(Y_tranche_mod) / 50 * 1000)  # En dBm (Puissance sur 50 ohms)
            ylabel = 'Amplitude (dBm)'
        else:
            raise ValueError("L'unité doit être 'Volts' ou 'dBm'.")

        # Affichage de la DSP
        plt.plot(f_tranche, Y_output)
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel(ylabel)
        plt.title('Densité Spectrale de Puissance (DSP)')
        plt.grid(True)
        plt.show()

        # Retourne les vecteurs fréquence et amplitude
        return f_tranche, Y_output


class Test :
    
    def print (self):
        print('test')