#!/usr/bin/env python3.10

import numpy as np
       
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
    def __init__(self, signal):
        self.signal = signal
        
    
    def DSP(signal,fe, type='Bi', unit='dBm'):
        N = len(signal)
        tfft = N*te
        te = 1/fe

        #Calcul de la FFT bilatérale
        Y_tranche = 1/N*(np.fft.fft(signal))

        #Calcul de la FFT Mono latérale
        Y_tranche_mono = np.concatenate((Y_tranche[0:1], 2*Y_tranche[1:int(N/2)]))
        Y_tranche_mod = np.abs(Y_tranche_mono)


        # Calcul de la Densité spectrale en Volt efficace sur RBW
        Y_eff_tranche = Y_tranche_mod/np.sqrt(2)

        # Calcul de la Densité spectrale en dBm sur RBW
        Y_dBm_tranche = 10*np.log10(np.square(Y_eff_tranche)/50*1000)

        # Calcul de la plage de fréquence pour la FFT
        f_tranche = np.arange(0, fe/2, fe/N)
        



class Test :
    
    def print (self):
        print('test')