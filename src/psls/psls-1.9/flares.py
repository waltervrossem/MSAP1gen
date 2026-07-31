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


def add_flares(LC, t_LC, t_flares, amp_flares, duration_flares, up_down, prot=0, incl=None, lat=None, lon=None):
    if prot == 0:
        for t0, amp0, dur0 in zip(t_flares, amp_flares, duration_flares):
            tmin = max([0, t0 - 3 * up_down * dur0])
            tmax = min([max(t_LC), t0 + 5 * dur0])
            t_LC_ = t_LC[(t_LC > tmin) & (t_LC < tmax)]
            LC[(t_LC > tmin) & (t_LC < tmax)] = LC[(t_LC > tmin) & (t_LC < tmax)] + flare(t0, amp0, dur0, up_down, t_LC_)
    else:
        t_LC_days = t_LC / 86400
        for t0, amp0, dur0, lat0, lon0 in zip(t_flares, amp_flares, duration_flares, lat, lon):
            tmin = max([0, t0 - 3 * up_down * dur0])
            tmax = min([max(t_LC), t0 + 5 * dur0])
            mask = (t_LC > tmin) & (t_LC < tmax)
            t_LC_ = t_LC[mask]
            t_LC_days_ = t_LC_days[mask]
            spot = spotintime.OneSpot(t_LC_days_, np.log(prot), 0, 2, lat0, lon0, None, None, 0)
            spot_profile = spot.dimming(incl, modul=0)
            if spot_profile.max() > 0:
                spot_profile = spot_profile / spot_profile.max()
            else:
                continue

            flare_LC0 = flare(t0, amp0, dur0, up_down, t_LC_)
            flare_LC = spot_profile * flare_LC0
            LC[(t_LC > tmin) & (t_LC < tmax)] = LC[(t_LC > tmin) & (t_LC < tmax)] + flare_LC

    return LC
