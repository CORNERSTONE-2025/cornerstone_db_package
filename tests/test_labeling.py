"""Parametric labeling: prediction label (YES/NO1..NO4) and stratification class on a small
synthetic sample."""
import pandas as pd
import pytest

from cornerstone.labeling import LabelConfig, issue_grid, label, threshold_flux

D = pd.Timestamp("2014-01-10", tz="UTC")


def day(n, hours=0):
    return D + pd.Timedelta(days=n, hours=hours)


SPANS = pd.DataFrame({"HARPNUM": [1, 2], "start": [day(-10), day(0)], "end": [day(10), day(0)]})


def flares(*items):
    return pd.DataFrame([{"HARPNUM": h, "t_peak": t, "flux": f} for h, t, f in items],
                        columns=["HARPNUM", "t_peak", "flux"])


def lab(fl, **cfg):
    cfg.setdefault("threshold", "M")
    out = label(SPANS, fl, LabelConfig(**cfg))
    return out.set_index(["t_X", "HARPNUM"])


def test_thresholds():
    assert threshold_flux("M") == 1e-5 and threshold_flux("x2") == 2e-4 and threshold_flux(3e-6) == 3e-6
    assert threshold_flux("C5.5") == pytest.approx(5.5e-6)
    with pytest.raises(ValueError):
        LabelConfig(step_hours=0)


def test_grid_alignment_and_observed_harps():
    g = issue_grid(day(0, 5), day(2), LabelConfig(step_hours=24))
    assert list(g) == [day(1), day(2)]                       # aligned to midnight (grid origin)
    out = label(SPANS, flares(), LabelConfig(step_hours=24))
    assert len(out[out.HARPNUM == 2]) == 1                   # observed only at day(0): start <= t_X <= end
    assert out.t_X.min() == day(-10) and out.t_X.max() == day(10)


def test_prediction_label_no_classes_unlimited_windows():
    fl = flares((1, day(-5), 2e-5),          # M2 five days before day 0
                (1, day(5), 3e-5))           # M3 five days after day 0
    out = lab(fl)
    assert out.loc[(day(-7), 1), "label"] == "NO2"          # only after (unlimited future)
    assert out.loc[(day(-6), 1), "label"] == "YES"          # flare at t_X + 24 h: inside (inclusive)
    assert out.loc[(day(-5), 1), "label"] == "YES"          # flare at t_X: inside the window (inclusive)
    assert out.loc[(day(0), 1), "label"] == "NO3"           # before and after
    assert out.loc[(day(4), 1), "label"] == "YES"           # M3 at day 5 within [t_X, t_X + 24 h]
    assert out.loc[(day(6), 1), "label"] == "NO4"           # only before
    assert out.loc[(day(0), 2), "label"] == "NO1"           # HARP without flares
    assert out.loc[(day(4), 1), "target"] == 1 and out.loc[(day(0), 1), "target"] == 0
    assert out.loc[(day(0), 1), ["n_before", "n_after", "n_window"]].tolist() == [1, 1, 0]


def test_bounded_stratification_window_reproduces_static_definition():
    fl = flares((1, day(-5), 2e-5), (1, day(5), 3e-5))
    out = lab(fl, strat_window_hours=24)
    assert out.loc[(day(0), 1), "label"] == "NO1"           # both flares outside [t_X - 24 h, t_X + 48 h]
    assert out.loc[(day(-4), 1), "label"] == "NO4"          # M2 within the 24 h before
    assert out.loc[(day(3), 1), "label"] == "NO2"           # M3 at day 5 in [t_X + 24 h, t_X + 48 h]


def test_threshold_and_stratification_class():
    fl = flares((1, day(0, 6), 3e-6),        # C3 in the window of day 0
                (1, day(2, 6), 2e-4))        # X2 two days later
    out = lab(fl, threshold="M", strat_threshold="C")
    r = out.loc[(day(0), 1)]
    assert r.label == "NO2" and r.target == 0                # M threshold: the C3 does not count
    assert r.strat_class == "C" and r.max_class == "C"       # C stratification: class of the window max
    assert out.loc[(day(2), 1), "strat_class"] == "X" and out.loc[(day(2), 1), "label"] == "YES"
    same = lab(fl, threshold="M")                            # S_s = S_p
    assert same.loc[(day(0), 1), "strat_class"] == "NO2"
    hi = lab(fl, threshold="C", strat_threshold="X")         # S_s > S_p allowed (user's choice)
    assert hi.loc[(day(0), 1), "label"] == "YES" and hi.loc[(day(0), 1), "strat_class"] == "NO2"


def test_flux_exactly_at_threshold_counts():
    out = lab(flares((1, day(0, 1), 1e-5)))
    assert out.loc[(day(0), 1), "label"] == "YES" and out.loc[(day(0), 1), "max_class"] == "M"


def test_flare_window_length():
    fl = flares((1, day(1, 12), 5e-5))
    assert lab(fl).loc[(day(0), 1), "label"] == "NO2"                              # 36 h > 24 h
    assert lab(fl, flare_window_hours=48).loc[(day(0), 1), "label"] == "YES"
    hourly = label(SPANS, fl, LabelConfig(threshold="M", step_hours=6, start=day(0), end=day(1)))
    assert len(hourly[hourly.HARPNUM == 1]) == 5
