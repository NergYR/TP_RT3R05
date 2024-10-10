#!/usr/bin/env python3.10

import numpy as np
import matplotlib.pyplot as plt
from scapy.all import *
import scapy.all
import scipy 



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
            
        self.nech = 1 
         
    def create_MP(self, amplitude, phase_origine=0):
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
            case ('PSK',4) :
                mapping_table = {(0,0) : amplitude * np.exp(1j*(phase_origine+np.pi/4)),
                                (0,1) : amplitude * np.exp(1j*(phase_origine+np.pi*3/4)),
                                (1,0) : amplitude * np.exp(1j*(phase_origine+np.pi*5/4)),
                                (1,1) : amplitude * np.exp(1j*(phase_origine+np.pi*7/4))}
            case ('PSK',16) :
                mapping_table = {(0,0,0,0) : -3-3j,
                                 (0,0,0,1) : -3-1j,
                                 (0,0,1,0) : -3+3j,
                                 (0,0,1,1) : -3+1j,
                                 (0,1,0,0) : -1-3j,
                                 (0,1,0,1) : -1-1j,
                                 (0,1,1,0) : -1+3j,
                                 (0,1,1,1) : -1+1j,
                                 (1,0,0,0) : 3-3j,
                                 (1,0,0,1) : 3-1j,
                                 (1,0,1,0) : 3+3j,
                                 (1,0,1,1) : 3+1j,
                                 (1,1,0,0) : 1-3j,
                                 (1,1,0,1) : 1-1j,
                                 (1,1,1,0) : 1+3j,
                                 (1,1,1,1) : 1+1j}   
            case _:
                mapping_table = None
                print(f'La modulation {self.nsymb}{self.modtype} n\'est pas implémentée')

        self.mapping_table = mapping_table
        return(mapping_table)

    def mapping(self, amplitude, phase_origine=0):
        self.mapping_table = self.create_MP(amplitude, phase_origine=0)
        symbs_mod=np.array([self.mapping_table[tuple(symb)] for symb in self.symbs_num])
        return(symbs_mod)
    
    def filtre_MF(self, symbols, upsampling, filtre_type='rectangular'):
        """
        Filtre de mise en forme pour les symboles de modulation.

        Paramètres :
        - symbols : vecteur des symboles de modulation à envoyer.
        - n_echantillons : nombre d'échantillons par symbole.
        - filtre_type : type de filtre ('rectangular', 'manchester', 'cosur').
        
        Retourne :
        - Un vecteur avec les échantillons.
        """
        self.nech = upsampling  # Attribuer le nombre d'échantillons à l'attribut d'instance
        samples = []

        if filtre_type == 'rectangular':
            for symbol in symbols:
                # Pour chaque symbole, ajouter n_echantillons de la même valeur
                samples.extend([symbol] * self.nech)
        elif filtre_type == 'manchester':
            # Implémentation du filtre Manchester (à compléter si nécessaire)
            raise NotImplementedError("Filtre Manchester non implémenté.")
        elif filtre_type == 'cosur':
            # Implémentation du filtre en cosinus surélevé (à compléter si nécessaire)
            raise NotImplementedError("Filtre en cosinus surélevé non implémenté.")
        else:
            raise ValueError("Type de filtre non reconnu. Utilisez 'rectangular', 'manchester' ou 'cosur'.")

        return np.array(samples)  # Retourner les échantillons sous forme de numpy array
    
    
    def downsample(self, signal, n, offset=0):
        
        
        """
        Réduit le taux d'échantillonnage d'un signal.

        Paramètres :
        - signal : signal à traiter.
        - n : facteur de réduction du taux d'échantillonnage.
        - offset: décalage initial (par défaut 0).

        Retourne :
        - Le signal échantillonné.
        """
        if self.symb_type == 'complexe':
            signal_down=np.array([], dtype=complex)
        else :
            signal_down=np.array([])
        for i in range(offset, len(signal), n):
            signal_down = np.append(signal_down,signal[i])
        return(signal_down) 
    
    
    def detection(self, symbs_rcv):
        constellation = np.array([val for val in self.mapping_table.values()])
        if self.symb_type == 'complexe':
            symbs_detect=[min(constellation, key=lambda symb_mod:abs(np.square(np.real(symbr)-
            np.real(symb_mod))+np.square(np.imag(symbr)-np.imag(symb_mod)))) for symbr in symbs_rcv]
        else :
            symbs_detect=[min(constellation, key=lambda symb_mod:abs(symbr-symb_mod)) for symbr in
            symbs_rcv]
        return(np.array(symbs_detect))
    
    
    def demapping(self, symbs_detect):
        demapping_table = {v: k for k, v in self.mapping_table.items()}
        symbs_num = np.array([demapping_table[symb] for symb in symbs_detect])
        return(symbs_num)
    
    def calcul_erreur_decodage(self, symbs_orig, symbs_detect):
        
        # Calcul de l'écart quadratique moyen
        mse = np.mean(np.square(symbs_orig - symbs_detect))
        return mse
    
    def upconv(self, env_complexe, f0, Te):
        t = np.arange(0, len(env_complexe)*Te, Te)
        reel = np.cos(2*np.pi*f0*t)
        im = np.sin(2*np.pi*f0*t)
        exp = reel+im*1j
        
        signal_analytique = env_complexe*exp
        modulated_signal = np.real(signal_analytique)
        
        
        
        return modulated_signal
    
    
    def downconv(self, mod_signal, f0, Te, symb_type='complexe'):
        t = np.arange(0, len(mod_signal)*Te, Te)
        reel = np.cos(2*np.pi*f0*t)
        im = np.sin(2*np.pi*f0*t)
        if symb_type == 'complexe':
            exp = reel-im*1j
            signal_analytique = exp*mod_signal
        else :
            signal_analytique = reel*mod_signal
        return signal_analytique
    
    
    def filtre_rcv(self, signal, fe=100, fc=10, type="butter", ordre=3):
        if type == "butter":
            b, a = scipy.signal.butter(ordre*2, 2 * fc / fe, 'low')
            signal_filtre = scipy.signal.filtfilt(b, a, signal)
        else: 
            raise ValueError("Le type de filtre doit être 'butter'.")
        return signal_filtre
    

# Définition de la classe Mesure
class Mesure:
    def __init__(self, signal):
        if not isinstance(signal, np.ndarray):
            raise ValueError("Le signal doit être un tableau numpy.")
        self.signal = signal

    def DSP(self, fe, type='Bi', unit='dBm'):
        Mysignal = self.signal
        N = Mysignal.shape[0]
        te = 1 / fe
        f_tranche = np.fft.fftfreq(N, d=te)

        # Calcul de la FFT
        Y_fft = np.fft.fft(Mysignal) / N

        if type == 'Bi':
            Y_tranche_mod = np.abs(Y_fft)
            f_tranche = np.fft.fftshift(f_tranche)
            Y_tranche_mod = np.fft.fftshift(Y_tranche_mod)
        elif type == 'mono':
            Y_tranche_mod = np.abs(Y_fft[:N // 2]) * 2
            f_tranche = f_tranche[:N // 2]
        else:
            raise ValueError("Le paramètre 'type' doit être 'Bi' ou 'mono'.")

        if unit == 'Volts':
            Y_output = Y_tranche_mod / np.sqrt(2)
            ylabel = 'Amplitude (Volts efficaces)'
        elif unit == 'dBm':
            Y_output = 10 * np.log10(np.square(Y_tranche_mod) / 50 * 1000)
            ylabel = 'Amplitude (dBm)'
        else:
            raise ValueError("L'unité doit être 'Volts' ou 'dBm'.")

        plt.plot(f_tranche, Y_output)
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel(ylabel)
        plt.title('Densité Spectrale de Puissance (DSP)')
        plt.grid(True)
        plt.show()

        return f_tranche, Y_output
    
    
    def plot_constellation(self, symbols, window_size=8, title="Diagramme de constellation"):
        """
        Affiche le diagramme de constellation pour les symboles fournis.
        
        Paramètres :
        - symbols : array numpy des symboles de modulation.
        - window_size : taille de la fenêtre carrée (par défaut 8).
        - title : titre de la fenêtre (par défaut "Diagramme de constellation").
        """
        plt.figure(figsize=(window_size, window_size))
        plt.scatter(np.real(symbols), np.imag(symbols), marker='o', color='blue')
        plt.title(title)
        plt.xlabel('Partie réelle des symboles de modulation')
        plt.ylabel('Partie imaginaire des symboles de modulation')
        plt.axhline(0, color='gray', lw=0.5, ls='--')
        plt.axvline(0, color='gray', lw=0.5, ls='--')
        plt.grid()
        plt.xlim(np.min(np.real(symbols)) - 1, np.max(np.real(symbols)) + 1)  # Ajuster les limites x
        plt.ylim(np.min(np.imag(symbols)) - 1, np.max(np.imag(symbols)) + 1)  # Ajuster les limites y
        plt.show()


class Source:
    def random(nb_bits):
        """
        Génère un vecteur de bits aléatoires suivant une loi binomiale.

        Paramètres :
        nb_bits : int : nombre de bits à générer

        Retourne :
        numpy.ndarray : tableau contenant la séquence binaire
        """
        # Génération de bits aléatoires avec une probabilité de 0.5 pour 0 ou 1
        bits = np.random.binomial(1, 0.5, nb_bits)
        return bits
    
    def frame_to_bits(frame):
        frame_dec =list(bytes(frame))
        frame_bin=[]
        for val in frame_dec:
            z=format(val, "08b")
            frame_bin += list(z)
            bits=np.array([int(x) for x in frame_bin])
        return(bits)

    def icmp(ip_dest, ip_src='192.168.1.1', mac_src='00:01:02:03:04:05', mac_dest='06:07:08:09:0A:0B', type='echo-request'):
        """
        Crée un paquet ICMP et retourne la séquence binaire correspondante.

        Paramètres :
        ip_dest : str : adresse IP de destination
        ip_src : str : adresse IP source (par défaut '192.168.1.1')
        mac_src : str : adresse MAC source (par défaut '00:01:02:03:04:05')
        mac_dest : str : adresse MAC destination (par défaut '06:07:08:09:0A:0B')
        type : str : type de paquet ICMP ('echo-request' ou 'echo-reply', par défaut 'echo-request')

        Retourne :
        numpy.ndarray : tableau contenant la séquence binaire du paquet ICMP
        """
        # Création du paquet ICMP
        if type == 'echo-request':
            packet = IP(src=ip_src, dst=ip_dest)/ICMP(type=8)  # Type 8 pour echo-request
        elif type == 'echo-reply':
            packet = IP(src=ip_src, dst=ip_dest)/ICMP(type=0)  # Type 0 pour echo-reply
        else:
            raise ValueError("Le type doit être 'echo-request' ou 'echo-reply'.")

        # Ajout des adresses MAC
        ether = Ether(src=mac_src, dst=mac_dest) / packet
        bits = Source.frame_to_bits(ether)
        
        return bits
    
    


class Canal: 
    @staticmethod 
    def awgn(signal, mean, std) :
        num_samples = len(signal)
        noise = np.random.normal(mean, std, size=num_samples)
        signal_bruite=signal+noise
        return(signal_bruite)
        


class Test :
    
    def print (self):
        print('test')
