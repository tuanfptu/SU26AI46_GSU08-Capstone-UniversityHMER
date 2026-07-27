"""Fast, deterministic-capable paper/camera augmentation for grayscale TAMER.

This intentionally uses OpenCV rather than a generative model. Labels remain
pixel-aligned and every transformation can be reproduced from a stored seed.
"""

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_io import read_grayscale


class DynamicPaperAugmentation:
    def __init__(
        self,
        background_dir: Optional[str] = None,
        real_background_probability: float = 0.70,
        mild_probability: float = 0.60,
        medium_probability: float = 0.30,
        max_height: int = 256,
        max_width: int = 1024,
    ) -> None:
        self.background_paths = []
        self.background_groups = {}
        if background_dir and Path(background_dir).exists():
            root = Path(background_dir)
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.background_paths.extend(root.rglob(suffix))
            for path in self.background_paths:
                relative = path.relative_to(root)
                group = relative.parts[0] if len(relative.parts) > 1 else "ungrouped"
                self.background_groups.setdefault(group, []).append(path)
        self.real_background_probability = real_background_probability
        self.mild_probability = mild_probability
        self.medium_probability = medium_probability
        self.max_height = max_height
        self.max_width = max_width

    def _severity(self, rng: np.random.RandomState, forced: Optional[str] = None) -> Tuple[str, float]:
        strengths = {"mild": 0.55, "medium": 1.0, "hard": 1.45}
        if forced is not None:
            if forced not in strengths:
                raise ValueError("severity must be mild, medium or hard")
            return forced, strengths[forced]
        draw = rng.rand()
        if draw < self.mild_probability:
            return "mild", strengths["mild"]
        if draw < self.mild_probability + self.medium_probability:
            return "medium", strengths["medium"]
        return "hard", strengths["hard"]

    @staticmethod
    def _paper_pattern(height: int, width: int, rng: np.random.RandomState) -> np.ndarray:
        base = float(rng.uniform(226, 250))
        paper = np.full((height, width), base, dtype=np.float32)

        # Real paper contains both fine sensor-visible grain and slowly varying
        # density from fibres. Single-scale Gaussian noise looks synthetic.
        paper += rng.normal(0, rng.uniform(0.7, 1.8), paper.shape).astype(np.float32)
        small_h = max(2, height // 24)
        small_w = max(2, width // 24)
        low_frequency = rng.normal(0, rng.uniform(2.0, 5.5), (small_h, small_w)).astype(np.float32)
        low_frequency = cv2.resize(low_frequency, (width, height), interpolation=cv2.INTER_CUBIC)
        low_frequency = cv2.GaussianBlur(low_frequency, (0, 0), max(2.0, min(height, width) / 35.0))
        paper += low_frequency

        # Sparse fibres and tiny manufacturing marks. They remain deliberately
        # faint so they cannot be mistaken for mathematical strokes.
        fibre_count = max(8, int(height * width / 18000))
        for _ in range(fibre_count):
            x = int(rng.randint(0, max(1, width)))
            y = int(rng.randint(0, max(1, height)))
            length = int(rng.randint(4, max(5, min(28, width // 8 + 1))))
            angle = float(rng.uniform(-0.25, 0.25))
            end = (min(width - 1, x + length), int(np.clip(y + math.sin(angle) * length, 0, height - 1)))
            cv2.line(paper, (x, y), end, base - rng.uniform(2, 7), 1, cv2.LINE_AA)

        kind = rng.choice(("a4", "ruled", "grid", "yellow", "worksheet"), p=(0.30, 0.25, 0.25, 0.10, 0.10))
        if kind == "yellow":
            paper -= rng.uniform(5, 18)
        spacing = int(rng.randint(22, 38))
        if kind in ("ruled", "grid"):
            offset = int(rng.randint(0, spacing))
            # Blue notebook rules become roughly 25-45 gray levels darker than
            # white paper after grayscale conversion on real phone photos.
            line_value = float(base - rng.uniform(35, 60))
            for y in range(offset, height, spacing):
                # Slightly different opacity per line is closer to printed paper.
                value = line_value + rng.uniform(-3, 3)
                thickness = 2 if height >= 150 and rng.rand() < 0.20 else 1
                cv2.line(paper, (0, y), (width - 1, y), value, thickness, cv2.LINE_AA)
        if kind == "grid":
            for x in range(int(rng.randint(0, spacing)), width, spacing):
                value = base - rng.uniform(30, 55)
                thickness = 2 if width >= 700 and rng.rand() < 0.12 else 1
                cv2.line(paper, (x, 0), (x, height - 1), value, thickness, cv2.LINE_AA)
        if kind == "ruled" and rng.rand() < 0.45:
            margin_x = int(rng.uniform(0.07, 0.16) * width)
            cv2.line(paper, (margin_x, 0), (margin_x, height - 1), base - rng.uniform(45, 70), 1, cv2.LINE_AA)
        if kind == "worksheet" and rng.rand() < 0.7:
            y = int(rng.randint(5, max(6, height // 5)))
            cv2.line(paper, (0, y), (width - 1, y), base - rng.uniform(20, 35), 1, cv2.LINE_AA)

        # Subtle fold and stain artefacts. Both are broad and low contrast.
        if rng.rand() < 0.13:
            fold_y = int(rng.randint(max(1, height // 8), max(2, 7 * height // 8)))
            fold = np.zeros_like(paper)
            cv2.line(fold, (0, fold_y), (width - 1, fold_y + int(rng.randint(-3, 4))), 1.0, 1, cv2.LINE_AA)
            fold = cv2.GaussianBlur(fold, (0, 0), rng.uniform(1.2, 3.5))
            paper -= fold * rng.uniform(5, 13)
        if rng.rand() < 0.10:
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            cx, cy = rng.uniform(0, width), rng.uniform(0, height)
            sx, sy = rng.uniform(width * 0.03, width * 0.12), rng.uniform(height * 0.05, height * 0.20)
            stain = np.exp(-(((xx - cx) / max(sx, 1)) ** 2 + ((yy - cy) / max(sy, 1)) ** 2) / 2.0)
            paper -= stain * rng.uniform(3, 11)
        return np.clip(paper, 0, 255).astype(np.uint8)

    def _real_background(self, height: int, width: int, rng: np.random.RandomState) -> Optional[np.ndarray]:
        if not self.background_paths or rng.rand() >= self.real_background_probability:
            return None
        if self.background_groups:
            group_names = sorted(self.background_groups)
            group = group_names[int(rng.randint(0, len(group_names)))]
            candidates = self.background_groups[group]
        else:
            candidates = self.background_paths
        path = candidates[int(rng.randint(0, len(candidates)))]
        image = read_grayscale(path)
        if image is None:
            return None
        h, w = image.shape
        scale = max(height / max(h, 1), width / max(w, 1))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = image.shape
        y = int(rng.randint(0, max(1, h - height + 1)))
        x = int(rng.randint(0, max(1, w - width + 1)))
        return image[y : y + height, x : x + width]

    @staticmethod
    def _ink_mask(clean: np.ndarray, rng: np.random.RandomState, strength: float) -> np.ndarray:
        mask = (255.0 - clean.astype(np.float32)) / 255.0
        if rng.rand() < 0.20 * strength:
            mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
        elif rng.rand() < 0.10 * strength:
            mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
        if rng.rand() < 0.25 * strength:
            sigma = float(rng.uniform(0.25, 0.85) * strength)
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        if rng.rand() < 0.30 * strength:
            modulation = cv2.GaussianBlur(rng.uniform(0.75, 1.15, mask.shape).astype(np.float32), (0, 0), 3.0)
            mask *= modulation
        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _perspective(image: np.ndarray, rng: np.random.RandomState, strength: float) -> np.ndarray:
        height, width = image.shape
        # 5-30 degree camera tilt is approximated by a bounded projective corner displacement.
        tilt_degrees = float(rng.triangular(5.0, 9.0, 30.0))
        fraction = min(0.18, math.tan(math.radians(tilt_degrees)) * 0.16) * strength
        dx = width * fraction
        dy = height * fraction * 0.55
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        direction = -1.0 if rng.rand() < 0.5 else 1.0
        dst = np.float32(
            [
                [rng.uniform(0, dx), rng.uniform(0, dy)],
                [width - 1 - rng.uniform(0, dx), rng.uniform(0, dy) * (1 if direction > 0 else 0.3)],
                [width - 1 - rng.uniform(0, dx), height - 1 - rng.uniform(0, dy)],
                [rng.uniform(0, dx), height - 1 - rng.uniform(0, dy) * (1 if direction < 0 else 0.3)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=245)

    @staticmethod
    def _page_on_desk(image: np.ndarray, rng: np.random.RandomState, strength: float) -> np.ndarray:
        """Photograph a paper crop with an occasional visible page/desk edge."""
        height, width = image.shape
        border_y = int(rng.uniform(0.10, 0.24) * height)
        border_x = int(rng.uniform(0.05, 0.14) * width)
        canvas_h = height + 2 * border_y
        canvas_w = width + 2 * border_x
        desk_base = float(rng.uniform(72, 176))
        desk = np.full((canvas_h, canvas_w), desk_base, dtype=np.float32)
        desk += rng.normal(0, rng.uniform(1.5, 4.0), desk.shape).astype(np.float32)
        desk_low = rng.normal(0, 7.0, (max(2, canvas_h // 30), max(2, canvas_w // 30))).astype(np.float32)
        desk += cv2.resize(desk_low, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
        desk = np.clip(desk, 0, 255).astype(np.uint8)

        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        jitter_x = border_x * min(0.85, 0.45 + 0.25 * strength)
        jitter_y = border_y * min(0.85, 0.45 + 0.25 * strength)
        dst = np.float32(
            [
                [border_x + rng.uniform(-jitter_x, jitter_x), border_y + rng.uniform(-jitter_y, jitter_y)],
                [border_x + width - 1 + rng.uniform(-jitter_x, jitter_x), border_y + rng.uniform(-jitter_y, jitter_y)],
                [border_x + width - 1 + rng.uniform(-jitter_x, jitter_x), border_y + height - 1 + rng.uniform(-jitter_y, jitter_y)],
                [border_x + rng.uniform(-jitter_x, jitter_x), border_y + height - 1 + rng.uniform(-jitter_y, jitter_y)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        page = cv2.warpPerspective(image, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=255)
        page_mask = cv2.warpPerspective(
            np.full_like(image, 255), matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=0
        ).astype(np.float32) / 255.0
        # Offset blurred mask creates a physical page-edge shadow on the desk.
        shadow_mask = cv2.GaussianBlur(page_mask, (0, 0), max(2.0, height / 30.0))
        shadow_mask = np.roll(shadow_mask, int(rng.uniform(2, max(3, border_y * 0.45))), axis=0)
        shadow_mask = np.roll(shadow_mask, int(rng.uniform(-border_x * 0.25, border_x * 0.25)), axis=1)
        desk_shadowed = desk.astype(np.float32) * (1.0 - shadow_mask * rng.uniform(0.10, 0.24))
        composite = page.astype(np.float32) * page_mask + desk_shadowed * (1.0 - page_mask)
        return np.clip(composite, 0, 255).astype(np.uint8)

    @staticmethod
    def _shadow_and_light(image: np.ndarray, rng: np.random.RandomState, strength: float) -> np.ndarray:
        height, width = image.shape
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        x_norm = xx / max(width - 1, 1)
        y_norm = yy / max(height - 1, 1)

        # Directional room/window illumination.
        angle = float(rng.uniform(0, 2 * np.pi))
        plane = np.cos(angle) * (x_norm - 0.5) + np.sin(angle) * (y_norm - 0.5)
        illumination = 1.0 + plane * rng.uniform(0.06, 0.18) * strength

        # A nearby lamp creates a broad radial bright region with realistic falloff.
        lamp_x = rng.uniform(-0.25, 1.25)
        lamp_y = rng.uniform(-0.35, 1.0)
        distance = ((x_norm - lamp_x) ** 2 + (y_norm - lamp_y) ** 2)
        lamp = np.exp(-distance / rng.uniform(0.20, 0.65))
        illumination += lamp * rng.uniform(0.02, 0.15) * strength

        # Soft cast shadow from a hand, phone or nearby object. It is a blurred
        # polygon/ellipse, not a simple global gradient.
        if rng.rand() < 0.55 * strength:
            shadow = np.zeros((height, width), dtype=np.float32)
            center = (int(rng.uniform(-0.1, 1.1) * width), int(rng.uniform(-0.1, 1.0) * height))
            axes = (int(rng.uniform(0.18, 0.55) * width), int(rng.uniform(0.12, 0.42) * height))
            cv2.ellipse(shadow, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
            sigma = max(5.0, min(height, width) * rng.uniform(0.05, 0.16))
            shadow = cv2.GaussianBlur(shadow, (0, 0), sigma)
            illumination *= 1.0 - shadow * rng.uniform(0.08, 0.25) * strength

        # Lens/phone vignette remains subtle; hard black corners are unrealistic.
        radial = np.sqrt(((x_norm - 0.5) / 0.72) ** 2 + ((y_norm - 0.5) / 0.82) ** 2)
        illumination *= 1.0 - np.clip(radial - 0.52, 0, 0.65) * rng.uniform(0.02, 0.12) * strength
        output = image.astype(np.float32) * illumination
        output = (output - 127.5) * rng.uniform(0.86, 1.14) + 127.5 + rng.uniform(-12, 12)
        gamma = float(rng.uniform(0.85, 1.15))
        output = 255.0 * np.power(np.clip(output, 0, 255) / 255.0, gamma)
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def _camera_degradation(image: np.ndarray, rng: np.random.RandomState, strength: float) -> np.ndarray:
        output = image
        if rng.rand() < 0.20 * strength:
            output = cv2.GaussianBlur(output, (0, 0), rng.uniform(0.3, 1.0) * strength)
        if rng.rand() < 0.10 * strength:
            kernel_size = int(rng.choice((3, 5, 7)))
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            output = cv2.filter2D(output, -1, kernel)
        if rng.rand() < 0.20 * strength:
            scale = float(rng.uniform(0.58, 0.90))
            small = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            output = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if rng.rand() < 0.35 * strength:
            sigma = float(rng.uniform(1.5, 7.0) * strength)
            noise = rng.normal(0, sigma, output.shape)
            output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.rand() < 0.40 * strength:
            quality = int(rng.randint(50, 96))
            ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                output = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        return output

    def __call__(
        self,
        clean: np.ndarray,
        seed: Optional[int] = None,
        severity: Optional[str] = None,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        if clean.ndim == 3:
            clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        clean = clean.astype(np.uint8)
        _, strength = self._severity(rng, forced=severity)
        pad_y = int(rng.randint(8, 25))
        pad_x = int(rng.randint(12, 45))
        clean = cv2.copyMakeBorder(clean, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)
        height, width = clean.shape
        paper = self._real_background(height, width, rng)
        using_real_background = paper is not None
        if not using_real_background:
            paper = self._paper_pattern(height, width, rng)
        mask = self._ink_mask(clean, rng, strength)
        ink_value = float(rng.choice((18, 35, 55, 75), p=(0.45, 0.30, 0.20, 0.05)))
        composite = paper.astype(np.float32) * (1.0 - mask) + ink_value * mask
        output = np.clip(composite, 0, 255).astype(np.uint8)
        # Most samples are expression crops. A minority retain a page/desk edge,
        # matching phone photos without shrinking the formula in every sample.
        if not using_real_background and rng.rand() < (0.08 + 0.12 * strength):
            output = self._page_on_desk(output, rng, strength)
        else:
            transform_strength = strength * (0.40 if using_real_background else 1.0)
            output = self._perspective(output, rng, transform_strength)
        effect_strength = strength * (0.45 if using_real_background else 1.0)
        output = self._shadow_and_light(output, rng, effect_strength)
        output = self._camera_degradation(output, rng, effect_strength)
        scale = min(self.max_height / output.shape[0], self.max_width / output.shape[1], 1.0)
        if scale < 1.0:
            output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return output
