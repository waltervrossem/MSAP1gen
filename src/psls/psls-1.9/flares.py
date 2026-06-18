import numpy as np

# F. Baudin (IAS)
# Usage
# LC_flare = add_flares(LC,t_LC,t_flares,amp_flares,duration_flares,up_down)
# Inputs :
# LC : courbe de lumière (en ppm, idéalement crée par PSLS)
# t_LC : timing de la courbe de lumière (en secondes)
# t_flares : timing des flares (en secondes, tableau de nflares éléments)
# amp_flares : amplitude des flares (en ppm, tableau de nflares éléments)
# duration_flares : durée des flares (en secondes, tableau de nflares éléments)
# up_down : ratio de la durée de montée du flux sur durée de descente du flux (scalaire)
# output :
# LC_flares : LC + flares


def flare(t0, amp0, dur0, up_down, t_LC_):
    LCflare = np.zeros(len(t_LC_))
    LCflare[t_LC_ <= t0] = amp0 * np.exp(-((t_LC_[t_LC_ <= t0] - t0) / (dur0 * up_down)) ** 2)
    LCflare[t_LC_ > t0] = amp0 * np.exp(-(t_LC_[t_LC_ > t0] - t0) / dur0)

    return LCflare


def add_flares(LC, t_LC, t_flares, amp_flares, duration_flares, up_down):
    for t0, amp0, dur0 in zip(t_flares, amp_flares, duration_flares):
        tmin = max([0, t0 - 3 * up_down * dur0])
        tmax = min([max(t_LC), t0 + 5 * dur0])
        t_LC_ = t_LC[(t_LC > tmin) & (t_LC < tmax)]
        LC[(t_LC > tmin) & (t_LC < tmax)] = LC[(t_LC > tmin) & (t_LC < tmax)] + flare(t0, amp0, dur0, up_down, t_LC_)

    return LC
