    
# Latex Vorlage
    \documentclass[journal,twoside,web]{ieeecolor}
    \usepackage{tmi}
    \usepackage{amsmath,amssymb,amsfonts}
    \usepackage{algorithmic}
    \usepackage{graphicx}
    \graphicspath{{images}}
    \usepackage{textcomp}
    \begin{document}

    % Struktur
    \section{Section}
    \subsection{Subsection}
    \subsubsection{Subsubsection}

    % Großer Anfangsbuchstabe
    \IEEEPARstart{T}{his}

    % Bilder einfügen
    \begin{figure}[!t]
        \centerline{\includegraphics[width=\columnwidth]{Architektur.png}}
        \caption{Magnetization as a function of applied field.
        It is good practice to explain the significance of the figure in the caption.}
        \label{fig:fig1}
    \end{figure}

    % Tabelle
    \begin{table}
        \caption{Units for Magnetic Properties}
        \label{table}
        \setlength{\tabcolsep}{3pt}
        \begin{tabular}{|p{25pt}|p{75pt}|p{115pt}|}
            \hline
            Symbol& 
            Quantity& 
            Conversion from Gaussian and \par CGS EMU to SI $^{\mathrm{a}}$ \\
            \hline
            $\Phi $& 
            magnetic flux& 
            1 Mx $\to  10^{-8}$ Wb $= 10^{-8}$ V$\cdot $s \\
            $B$& 
            magnetic flux density, \par magnetic induction& 
            1 G $\to  10^{-4}$ T $= 10^{-4}$ Wb/m$^{2}$ \\
            $H$& 
            magnetic field strength& 
            1 Oe $\to  10^{3}/(4\pi )$ A/m \\
            $m$& 
            magnetic moment& 
            1 erg/G $=$ 1 emu \par $\to 10^{-3}$ A$\cdot $m$^{2} = 10^{-3}$ J/T \\
            $M$& 
            magnetization& 
            1 erg/(G$\cdot $cm$^{3}) =$ 1 emu/cm$^{3}$ \par $\to 10^{3}$ A/m \\
            4$\pi M$& 
            magnetization& 
            1 G $\to  10^{3}/(4\pi )$ A/m \\
            $\sigma $& 
            specific magnetization& 
            1 erg/(G$\cdot $g) $=$ 1 emu/g $\to $ 1 A$\cdot $m$^{2}$/kg \\
            $j$& 
            magnetic dipole \par moment& 
            1 erg/G $=$ 1 emu \par $\to 4\pi \times  10^{-10}$ Wb$\cdot $m \\
            $J$& 
            magnetic polarization& 
            1 erg/(G$\cdot $cm$^{3}) =$ 1 emu/cm$^{3}$ \par $\to 4\pi \times  10^{-4}$ T \\
            $\chi , \kappa $& 
            susceptibility& 
            1 $\to  4\pi $ \\
            $\chi_{\rho }$& 
            mass susceptibility& 
            1 cm$^{3}$/g $\to  4\pi \times  10^{-3}$ m$^{3}$/kg \\
            $\mu $& 
            permeability& 
            1 $\to  4\pi \times  10^{-7}$ H/m \par $= 4\pi \times  10^{-7}$ Wb/(A$\cdot $m) \\
            $\mu_{r}$& 
            relative permeability& 
            $\mu \to \mu_{r}$ \\
            $w, W$& 
            energy density& 
            1 erg/cm$^{3} \to  10^{-1}$ J/m$^{3}$ \\
            $N, D$& 
            demagnetizing factor& 
            1 $\to  1/(4\pi )$ \\
            \hline
            \multicolumn{3}{p{251pt}}{Vertical lines are optional in tables. Statements that serve as captions for 
            the entire table do not need footnote letters. }\\
            \multicolumn{3}{p{251pt}}{$^{\mathrm{a}}$Gaussian units are the same as cg emu for magnetostatics; Mx 
            $=$ maxwell, G $=$ gauss, Oe $=$ oersted; Wb $=$ weber, V $=$ volt, s $=$ 
            second, T $=$ tesla, m $=$ meter, A $=$ ampere, J $=$ joule, kg $=$ 
            kilogram, H $=$ henry.}
        \end{tabular}
        \label{tab:tab1}
    \end{table}
    \end{document}