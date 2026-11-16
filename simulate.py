"""Minimal Monte Carlo example for a weak laser-target echo in atmosphere.

The example intentionally uses only NumPy and the Python standard library.
It generates:
  * a scenario-level CSV containing theoretical and Monte Carlo SNR;
  * an NPZ file containing one representative H0/H1 waveform pair;
  * SVG line charts that can be opened in a browser.

Run:
    python simulate.py
"""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 2.99792458e8


# -----------------------------
# 1. System and detector model
# -----------------------------


@dataclass(frozen=True)
class LaserSystem:
    wavelength_m: float = 1064e-9
    pulse_energy_j: float = 100e-6
    pulse_fwhm_s: float = 20e-9
    divergence_rad: float = 0.30e-3
    tx_efficiency: float = 0.80
    rx_efficiency: float = 0.80
    aperture_diameter_m: float = 0.15
    optical_filter_bandwidth_m: float = 2e-9  # 2 nm
    fov_half_angle_rad: float = 0.5e-3
    quantum_efficiency: float = 0.70
    dark_count_rate_cps: float = 1_000.0
    read_noise_e: float = 12.0
    pointing_efficiency: float = 0.90
    angstrom_exponent: float = 0.50
    observation_window_s: float = 100e-9
    samples_per_window: int = 200


@dataclass(frozen=True)
class TargetScene:
    slant_range_m: float
    visibility_km: float
    turbulence_sigma_ln: float
    target_reflectivity: float = 0.20
    target_projected_area_m2: float = 0.05
    incidence_angle_rad: float = 0.0
    # W / (m^2 sr m); 1e8 is 0.1 W / (m^2 sr nm).
    background_radiance: float = 1.0e8


def photon_energy_j(system: LaserSystem) -> float:
    return PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / system.wavelength_m


def received_mean_energy(system: LaserSystem, scene: TargetScene) -> tuple[float, float, float]:
    """Return received target pulse energy, background power, and extinction.

    The target equation assumes a small Lambertian diffuse target:

        Er = Et * eta_t * eta_r * T^2 * eta_p
             * 4 * rho * A_target * cos(theta) * A_rx
             / (pi^2 * divergence^2 * R^4)

    ``T = exp(-gamma R)`` uses a Koschmieder extinction approximation whose
    visibility is specified at the laser wavelength. The wavelength adjustment
    is only a simple Angstrom-law approximation and should be replaced by
    MODTRAN/Py6S or measured extinction coefficients in a full study.
    """

    range_m = scene.slant_range_m
    visibility_m = scene.visibility_km * 1_000.0
    extinction_550_per_m = 3.912 / visibility_m
    wavelength_ratio = system.wavelength_m / 550e-9
    extinction_per_m = extinction_550_per_m * wavelength_ratio ** (-system.angstrom_exponent)

    one_way_transmittance = math.exp(-extinction_per_m * range_m)
    two_way_transmittance = one_way_transmittance**2
    aperture_area = math.pi * system.aperture_diameter_m**2 / 4.0
    cosine_factor = math.cos(scene.incidence_angle_rad)

    received_energy = (
        system.pulse_energy_j
        * system.tx_efficiency
        * system.rx_efficiency
        * system.pointing_efficiency
        * two_way_transmittance
        * 4.0
        * scene.target_reflectivity
        * scene.target_projected_area_m2
        * cosine_factor
        * aperture_area
        / (math.pi**2 * system.divergence_rad**2 * range_m**4)
    )

    fov_solid_angle = 2.0 * math.pi * (1.0 - math.cos(system.fov_half_angle_rad))
    background_power = (
        scene.background_radiance
        * system.optical_filter_bandwidth_m
        * aperture_area
        * fov_solid_angle
        * system.rx_efficiency
    )
    return received_energy, background_power, extinction_per_m


def pulse_template(system: LaserSystem) -> np.ndarray:
    """Gaussian pulse shape normalized so that its samples sum to one."""

    sample_count = system.samples_per_window
    sample_time = system.observation_window_s / sample_count
    time = (np.arange(sample_count) - (sample_count - 1) / 2.0) * sample_time
    sigma = system.pulse_fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    template = np.exp(-0.5 * (time / sigma) ** 2)
    return template / template.sum()


def make_waveform_pair(
    system: LaserSystem,
    scene: TargetScene,
    rng: np.random.Generator,
    template: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Create one noise-only waveform (H0) and one echo-plus-noise (H1)."""

    mean_energy, background_power, _ = received_mean_energy(system, scene)
    photon_energy = photon_energy_j(system)

    # Mean-one lognormal model: E[x] = 1; larger sigma_ln means stronger fading.
    sigma = scene.turbulence_sigma_ln
    fade = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)

    signal_electrons = (
        system.quantum_efficiency * mean_energy * fade / photon_energy
    )
    background_electrons = (
        system.quantum_efficiency
        * background_power
        * system.observation_window_s
        / photon_energy
    )
    dark_electrons = (
        system.dark_count_rate_cps * system.observation_window_s
    )

    background_mean = (
        background_electrons + dark_electrons
    ) / system.samples_per_window
    signal_mean = signal_electrons * template

    h0 = rng.poisson(background_mean, size=system.samples_per_window).astype(float)
    h1 = rng.poisson(signal_mean + background_mean).astype(float)
    h0 += rng.normal(0.0, system.read_noise_e, size=h0.size)
    h1 += rng.normal(0.0, system.read_noise_e, size=h1.size)

    return {
        "h0": h0,
        "h1": h1,
        "fade": float(fade),
        "mean_received_energy_j": mean_energy,
        "mean_signal_electrons": float(signal_electrons),
        "background_electrons": float(background_electrons),
        "dark_electrons": float(dark_electrons),
    }


def matched_filter_metrics(
    system: LaserSystem,
    scene: TargetScene,
    template: np.ndarray,
    realizations: int,
    seed: int,
) -> dict[str, float | np.ndarray]:
    """Run a Monte Carlo simulation for one scene."""

    rng = np.random.default_rng(seed)
    normalized_template = template / np.linalg.norm(template)
    z0 = np.empty(realizations)
    z1 = np.empty(realizations)
    mean_signal_electrons = np.empty(realizations)
    fades = np.empty(realizations)

    for index in range(realizations):
        pair = make_waveform_pair(
            system=system,
            scene=scene,
            rng=rng,
            template=template,
        )
        z0[index] = np.dot(normalized_template, pair["h0"])
        z1[index] = np.dot(normalized_template, pair["h1"])
        mean_signal_electrons[index] = pair["mean_signal_electrons"]
        fades[index] = pair["fade"]

    threshold = float(np.quantile(z0, 0.99))
    detection_probability = float(np.mean(z1 > threshold))
    noise_std_h0 = float(np.std(z0, ddof=1))
    noise_std_h1 = float(np.std(z1, ddof=1))
    # Analytical SNR for the same matched filter, evaluated at mean fading.
    mean_energy, background_power, extinction_per_m = received_mean_energy(
        system=system,
        scene=scene,
    )
    photon_energy = photon_energy_j(system)
    signal_electrons = system.quantum_efficiency * mean_energy / photon_energy
    background_electrons = (
        system.quantum_efficiency
        * background_power
        * system.observation_window_s
        / photon_energy
    )
    dark_electrons = system.dark_count_rate_cps * system.observation_window_s
    bg_per_sample = (
        background_electrons + dark_electrons
    ) / system.samples_per_window
    h = normalized_template
    signal_mean = signal_electrons * template
    signal_projection = float(np.dot(h, signal_mean))
    noise_variance_projection = float(
        np.sum(h**2 * (bg_per_sample + signal_mean + system.read_noise_e**2))
    )
    theoretical_snr = 20.0 * math.log10(
        max(signal_projection, np.finfo(float).tiny)
        / math.sqrt(max(noise_variance_projection, np.finfo(float).tiny))
    )
    theoretical_snr_linear = signal_projection / math.sqrt(
        max(noise_variance_projection, np.finfo(float).tiny)
    )
    monte_carlo_signal_projection = float(
        np.dot(h, np.mean(mean_signal_electrons) * template)
    )
    monte_carlo_snr = 20.0 * math.log10(
        max(monte_carlo_signal_projection, np.finfo(float).tiny)
        / max(noise_std_h1, np.finfo(float).tiny)
    )
    snr_by_sampled_fade = 20.0 * np.log10(
        np.maximum(fades * theoretical_snr_linear, np.finfo(float).tiny)
    )

    return {
        "mean_received_energy_j": mean_energy,
        "mean_signal_electrons": float(np.mean(mean_signal_electrons)),
        "background_electrons": background_electrons,
        "extinction_per_m": extinction_per_m,
        "one_way_transmittance": math.exp(-extinction_per_m * scene.slant_range_m),
        "scintillation_index": float(np.exp(scene.turbulence_sigma_ln**2) - 1.0),
        "mean_fade": float(np.mean(fades)),
        "snr_theory_db": float(theoretical_snr),
        "snr_mc_mean_db": float(monte_carlo_snr),
        "snr_mc_p10_db": float(np.quantile(snr_by_sampled_fade, 0.10)),
        "snr_mc_p90_db": float(np.quantile(snr_by_sampled_fade, 0.90)),
        "threshold_at_pfa_1pct_e": threshold,
        "detection_probability_at_pfa_1pct": detection_probability,
        "z0": z0,
        "z1": z1,
    }


# -----------------------------
# 2. Small dependency-free SVG
# -----------------------------


def write_line_svg(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
    show_markers: bool = True,
) -> None:
    width, height = 920, 540
    left, right, top, bottom = 90, 35, 70, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in series.values() for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0
    y_padding = 0.08 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ["#0b6e99", "#d1495b", "#2a9d8f", "#6a4c93", "#e76f51"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="540" '
        'viewBox="0 0 920 540">',
        '<rect width="920" height="540" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" '
        'font-family="Arial,Microsoft YaHei,sans-serif" font-size="22" '
        f'font-weight="700">{html.escape(title)}</text>',
    ]

    for tick in range(6):
        fraction = tick / 5.0
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x = sx(x_value)
        y = sy(y_value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#e6e8eb" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e6e8eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            'font-family="Arial,Microsoft YaHei,sans-serif" font-size="12" '
            f'fill="#444">{x_value:.1f}</text>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="Arial,Microsoft YaHei,sans-serif" font-size="12" '
            f'fill="#444">{y_value:.1f}</text>'
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
            'stroke="#333" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            'stroke="#333" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 24}" '
            'text-anchor="middle" font-family="Arial,Microsoft YaHei,sans-serif" '
            f'font-size="15">{html.escape(x_label)}</text>',
            f'<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-family="Arial,Microsoft YaHei,sans-serif" '
            f'font-size="15">{html.escape(y_label)}</text>',
        ]
    )

    for series_index, (label, points) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        coordinates = " ".join(
            f"{sx(x):.2f},{sy(y):.2f}" for x, y in sorted(points)
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        if show_markers:
            for x, y in sorted(points):
                parts.append(
                    f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>'
                )
        legend_x = left + 18
        legend_y = top + 16 + series_index * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 30}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{legend_x + 38}" y="{legend_y + 5}" '
            'font-family="Arial,Microsoft YaHei,sans-serif" font-size="13" '
            f'fill="#333">{html.escape(label)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# -----------------------------
# 3. Scenario sweep and output
# -----------------------------


def main() -> None:
    system = LaserSystem()
    template = pulse_template(system)
    output_dir = Path(__file__).resolve().parent
    realizations = 300
    seed = 20261116

    ranges_km = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    visibilities_km = [2.0, 5.0, 10.0, 20.0]
    turbulence_levels = {
        "weak": 0.10,
        "moderate": 0.30,
        "strong": 0.55,
    }

    rows: list[dict[str, float | int | str]] = []
    scenario_number = 0
    for visibility_km in visibilities_km:
        for turbulence_name, sigma_ln in turbulence_levels.items():
            for range_km in ranges_km:
                scenario_number += 1
                scene = TargetScene(
                    slant_range_m=range_km * 1_000.0,
                    visibility_km=visibility_km,
                    turbulence_sigma_ln=sigma_ln,
                )
                metrics = matched_filter_metrics(
                    system=system,
                    scene=scene,
                    template=template,
                    realizations=realizations,
                    seed=seed + scenario_number,
                )
                rows.append(
                    {
                        "scenario_id": scenario_number,
                        "range_km": range_km,
                        "visibility_km": visibility_km,
                        "turbulence": turbulence_name,
                        "turbulence_sigma_ln": sigma_ln,
                        "scintillation_index": metrics["scintillation_index"],
                        "extinction_per_km": metrics["extinction_per_m"] * 1_000.0,
                        "one_way_transmittance": metrics["one_way_transmittance"],
                        "mean_received_energy_j": metrics[
                            "mean_received_energy_j"
                        ],
                        "mean_signal_electrons": metrics["mean_signal_electrons"],
                        "background_electrons": metrics["background_electrons"],
                        "snr_theory_db": metrics["snr_theory_db"],
                        "snr_mc_mean_db": metrics["snr_mc_mean_db"],
                        "snr_mc_p10_db": metrics["snr_mc_p10_db"],
                        "snr_mc_p90_db": metrics["snr_mc_p90_db"],
                        "detection_probability_at_pfa_1pct": metrics[
                            "detection_probability_at_pfa_1pct"
                        ],
                    }
                )

    csv_path = output_dir / "scenario_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Representative weak-echo pair: 8 km, 10 km visibility, moderate fading.
    representative_scene = TargetScene(
        slant_range_m=8_000.0,
        visibility_km=10.0,
        turbulence_sigma_ln=0.30,
    )
    representative_rng = np.random.default_rng(seed + 50_000)
    pair = make_waveform_pair(
        system=system,
        scene=representative_scene,
        rng=representative_rng,
        template=template,
    )
    sample_time = system.observation_window_s / system.samples_per_window
    time_relative_to_arrival = (
        np.arange(system.samples_per_window) - (system.samples_per_window - 1) / 2.0
    ) * sample_time
    np.savez_compressed(
        output_dir / "representative_waveforms.npz",
        time_relative_to_arrival_s=time_relative_to_arrival,
        pulse_template=template,
        h0_noise_only_e=pair["h0"],
        h1_echo_plus_noise_e=pair["h1"],
        scene_json=np.array(json.dumps(asdict(representative_scene))),
        system_json=np.array(json.dumps(asdict(system))),
    )
    waveform_series = {
        "H0：纯噪声": [
            (float(t * 1e9), float(value))
            for t, value in zip(time_relative_to_arrival, pair["h0"])
        ],
        "H1：弱回波 + 噪声": [
            (float(t * 1e9), float(value))
            for t, value in zip(time_relative_to_arrival, pair["h1"])
        ],
        "无噪声目标信号": [
            (float(t * 1e9), float(value * pair["mean_signal_electrons"]))
            for t, value in zip(time_relative_to_arrival, template)
        ],
    }
    write_line_svg(
        output_dir / "representative_waveform.svg",
        waveform_series,
        "8 km、10 km 能见度、中等湍流下的代表性弱回波",
        "相对目标到达时刻 / ns",
        "光电子数 / e−",
        show_markers=False,
    )

    # SNR versus range for visibility values at moderate turbulence.
    plot_floor_db = -80.0
    range_series: dict[str, list[tuple[float, float]]] = {}
    for visibility_km in visibilities_km:
        selected = [
            row
            for row in rows
            if row["visibility_km"] == visibility_km
            and row["turbulence"] == "moderate"
        ]
        range_series[f"能见度 {visibility_km:g} km"] = [
            (
                float(row["range_km"]),
                max(plot_floor_db, float(row["snr_mc_mean_db"])),
            )
            for row in selected
        ]
    write_line_svg(
        output_dir / "snr_vs_range.svg",
        range_series,
        "复杂大气环境下弱目标回波 SNR（中等湍流）",
        "目标距离 R / km",
        "匹配滤波 SNR / dB（显示下限 -80 dB）",
    )

    # SNR versus visibility at 5 km for all turbulence levels.
    visibility_series: dict[str, list[tuple[float, float]]] = {}
    for turbulence_name in turbulence_levels:
        selected = [
            row
            for row in rows
            if row["range_km"] == 5.0 and row["turbulence"] == turbulence_name
        ]
        visibility_series[f"湍流：{turbulence_name}"] = [
            (
                float(row["visibility_km"]),
                max(plot_floor_db, float(row["snr_mc_mean_db"])),
            )
            for row in selected
        ]
    write_line_svg(
        output_dir / "snr_vs_visibility.svg",
        visibility_series,
        "5 km 距离下 SNR 随能见度与湍流变化",
        "能见度 V / km",
        "匹配滤波 SNR / dB（显示下限 -80 dB）",
    )

    representative_row = next(
        row
        for row in rows
        if row["range_km"] == 8.0
        and row["visibility_km"] == 10.0
        and row["turbulence"] == "moderate"
    )
    summary = {
        "description": "复杂大气环境下激光弱目标回波 Monte Carlo 仿真示例",
        "realizations_per_scenario": realizations,
        "scenario_count": len(rows),
        "representative_scene": representative_row,
        "model_notes": [
            "大气衰减采用 Beer-Lambert 与 Koschmieder 能见度近似。",
            "湍流采用均值为 1 的对数正态随机透射因子。",
            "目标采用小型朗伯漫反射平面目标。",
            "波形同时包含信号相关散粒噪声、背景光、暗计数和读出噪声。",
            "SNR 由归一化匹配滤波后的信号对比度给出。",
            "本数据是物理模型仿真结果，不代表实际外场实验数据。",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Generated {len(rows)} scenarios in {output_dir}")
    print(
        "Representative 8 km / 10 km visibility / moderate turbulence: "
        f"MC SNR = {representative_row['snr_mc_mean_db']:.2f} dB"
    )


if __name__ == "__main__":
    main()
