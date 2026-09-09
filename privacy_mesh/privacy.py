from opacus.accountants import RDPAccountant


def calibrate_sigma(target_eps, delta, sample_rate, steps):
    if sample_rate <= 0 or steps <= 0:
        raise ValueError("sample_rate and steps must be positive")
    lo, hi = 0.05, 30.0

    def eps_for(sigma):
        acct = RDPAccountant()
        for _ in range(steps):
            acct.step(noise_multiplier=sigma, sample_rate=sample_rate)
        return acct.get_epsilon(delta=delta)

    if eps_for(hi) > target_eps:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if eps_for(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    eps = eps_for(sigma)
    if eps > target_eps * 1.001:
        sigma *= 1.02
    return round(float(sigma), 4)
