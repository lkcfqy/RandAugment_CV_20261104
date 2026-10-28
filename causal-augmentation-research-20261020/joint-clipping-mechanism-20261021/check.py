"""One-shot exact-rational controlled mechanism check. Standard library only."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"


def clip(x):
    return min(F(1), max(F(0), x))


def encoded_number(x):
    x = F(x)
    return {"exact": str(x), "decimal": float(x)}


def sign(x):
    return 1 if x >= 0 else -1


def interpolate(x, knots):
    for (a, ya), (b, yb) in zip(knots, knots[1:]):
        if a <= x <= b:
            return ya + (x - a) * (yb - ya) / (b - a)
    raise ValueError("outside the prescribed camera response")


def main():
    started = time.perf_counter()
    protocol_bytes = PROTOCOL.read_bytes()
    p = json.loads(protocol_bytes)
    high, low = F(p["red_high"]), F(p["red_low"])
    blue_high, blue_low = F(p["blue_high"]), F(p["blue_low"])
    green = F(p["green"])
    e, c = F(p["exposure_gain"]), F(p["illumination_red_gain"])
    gains = list(map(F, p["exact_grid"]["gains"]))
    identity_checks = gamma_checks = 0
    joint_only_at_white = joint_only_strictly_truncated = 0
    for i in range(101):
        x = F(i, 100)
        for a in gains:
            for b in gains:
                y0, y1, y2, y12 = x, clip(a*x), clip(b*x), clip(a*b*x)
                predicted = clip(y1*y2/y0) if y0 else F(0)
                assert predicted == y12
                assert clip(a*clip(b*x)) == clip(b*clip(a*x)) == y12
                mask = y1 < 1 and y2 < 1 and a*b*x >= 1
                strict_mask = y1 < 1 and y2 < 1 and a*b*x > 1
                assert mask == (F(1)/(a*b) <= x < F(1)/max(a,b))
                assert strict_mask == (F(1)/(a*b) < x < F(1)/max(a,b))
                joint_only_at_white += mask
                joint_only_strictly_truncated += strict_mask
                identity_checks += 1
                for gamma in p["exact_grid"]["pure_power_exponents"]:
                    z0, z1, z2 = y0**gamma, y1**gamma, y2**gamma
                    pred_gamma = clip(z1*z2/z0) if z0 else F(0)
                    assert pred_gamma == y12**gamma
                    gamma_checks += 1

    def image(y, n):
        red = (high, low) if y == 1 else (low, high)
        blue = (blue_high, blue_low) if y*n == 1 else (blue_low, blue_high)
        return [(red[j], green, blue[j]) for j in range(2)]

    def transform(pixels, exposure, red_gain, truncate=True, reverse=False):
        def apply(pxs, mult):
            return [tuple(clip(v*m) if truncate else v*m for v,m in zip(px,mult)) for px in pxs]
        operations = [(exposure,)*3, (red_gain,F(1),F(1))]
        if reverse:
            operations.reverse()
        for op in operations:
            pixels = apply(pixels, op)
        return pixels

    def risk_table(exposure, red_gain, truncate=True):
        environments = {"source":(F(1),F(1)), "exposure_only":(exposure,F(1)),
                        "illumination_only":(F(1),red_gain), "joint":(exposure,red_gain)}
        risks, states = {}, []
        for name, (a,b) in environments.items():
            risk_c = risk_b = F(0)
            for y in (-1,1):
                for n, prob_n in ((1,F(9,10)),(-1,F(1,10))):
                    weight = F(1,2)*prob_n
                    pixels = transform(image(y,n),a,b,truncate)
                    assert pixels == transform(image(y,n),a,b,truncate,reverse=True)
                    pred_c = sign(pixels[0][0]-pixels[1][0])
                    pred_b = sign(pixels[0][2]-pixels[1][2])
                    risk_c += weight*(pred_c != y)
                    risk_b += weight*(pred_b != y)
                    states.append({"environment":name,"unit_Y":y,"unit_N":n,
                                   "probability":encoded_number(weight),
                                   "rgb_patches":[[str(v) for v in px] for px in pixels],
                                   "prediction_color":pred_c,"prediction_blue":pred_b})
            risks[name] = {"h_color":risk_c,"h_blue":risk_b}
        return risks, states

    main_risks, states = risk_table(e,c)
    no_clip_risks, _ = risk_table(e,c,False)
    lower_gain_risks, _ = risk_table(F(6,5),F(6,5))
    for name, risks in main_risks.items():
        assert risks["h_color"] == (F(1,2) if name == "joint" else F(0))
        assert risks["h_blue"] == F(1,10)
    for control in (no_clip_risks,lower_gain_risks):
        for risks in control.values():
            assert risks == {"h_color":F(0),"h_blue":F(1,10)}
    regret = main_risks["joint"]["h_color"]-main_risks["joint"]["h_blue"]
    assert regret == F(2,5)

    phase = []
    for k in (F(1),F(3,2),F(20,13),F(8,5),F(20,11),F(49,25)):
        actual = clip(k*high)-clip(k*low)
        predicted = k*(high-low) if k <= 1/high else (1-k*low if k < 1/low else F(0))
        assert actual == predicted
        phase.append({"product_gain":encoded_number(k),"red_margin":encoded_number(actual),
                      "red_patches_at_white":int(k*high>=1)+int(k*low>=1),
                      "color_risk":encoded_number(F(1,2) if actual==0 else F(0))})

    # Same observed source and singles, known gains, fixed but unknown response.
    knots = [(F(0),F(0)),(F(3,10),F(11,20)),(F(21,50),F(77,100)),
             (F(147,250),F(9,10)),(F(1),F(1))]
    assert all(a < b and ya < yb for (a,ya),(b,yb) in zip(knots,knots[1:]))
    x_a, x_b = F(11,20), F(3,10)
    observed_a = [x_a,clip(e*x_a),clip(c*x_a)]
    observed_b = [interpolate(clip(g*x_b),knots) for g in (F(1),e,c)]
    assert observed_a == observed_b == [F(11,20),F(77,100),F(77,100)]
    joint_a = clip(e*c*x_a)
    joint_b = interpolate(clip(e*c*x_b),knots)
    assert joint_a == F(1) and joint_b == F(9,10)
    assert e*c*x_a > 1 and e*c*x_b < 1

    def encode_risks(table):
        return {env:{h:encoded_number(v) for h,v in risks.items()} for env,risks in table.items()}

    output = {
        "status":"all_prespecified_exact_checks_passed",
        "finished_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":hashlib.sha256(protocol_bytes).hexdigest(),
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine(),
                     "device":"CPU","dependencies":"Python standard library only","elapsed_seconds":time.perf_counter()-started,
                     "peak_rss_native_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "peak_rss_units":"bytes on macOS; KiB on Linux"},
        "exact_grid":{"identity_and_commutation_cases":identity_checks,"pure_power_cases":gamma_checks,
                      "joint_only_at_white_cases":joint_only_at_white,
                      "joint_only_strictly_truncated_cases":joint_only_strictly_truncated},
        "main_exact_population_risks":encode_risks(main_risks),
        "joint_selection_regret":encoded_number(regret),
        "without_truncation_mathematical_control":encode_risks(no_clip_risks),
        "lower_gain_6_over_5_control":encode_risks(lower_gain_risks),
        "thresholds":{"high_first_reaches_white_product_gain":encoded_number(1/high),
                      "low_reaches_white_and_color_cue_collapses_product_gain":encoded_number(1/low),
                      "joint_only_at_white_interval_lower_inclusive":encoded_number(1/(e*c)),
                      "joint_only_at_white_interval_upper_exclusive":encoded_number(1/max(e,c))},
        "fixed_margin_phase_checks":phase,
        "unknown_fixed_monotone_response_counterexample":{
            "shared_observed_source_and_singles":[encoded_number(v) for v in observed_a],
            "world_A":"identity response, latent source 11/20, joint raw signal 539/500 > 1",
            "world_B":"fixed strictly increasing piecewise-linear response, latent source 3/10, joint raw signal 147/250 < 1",
            "world_B_response_knots":[[str(v) for v in xy] for xy in knots],
            "world_A_joint":encoded_number(joint_a),"world_B_joint":encoded_number(joint_b)},
        "enumerated_states_not_independent_empirical_samples":states,
        "scope":"Controlled synthetic exact expectation, not an estimated accuracy or trained-policy comparison"
    }
    result_path = ROOT / "results.json"
    with result_path.open("x") as f:
        json.dump(output,f,indent=2,ensure_ascii=False)
        f.write("\n")
    print(json.dumps({"status":output["status"],"result_path":str(result_path),
                      "exact_grid":output["exact_grid"],"main_exact_population_risks":output["main_exact_population_risks"],
                      "elapsed_seconds":output["execution"]["elapsed_seconds"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
