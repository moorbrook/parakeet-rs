#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools>=9.0", "numpy>=2.0"]
# ///
"""Pin the native RNNT decode loop to the Core ML models it replaces.

The worker runs the Parakeet Unified prediction network (embedding + two LSTM
layers) and the joint decision head in native code, so the greedy decode loop
costs no Core ML dispatches. That is only safe if the native arithmetic follows
the compiled programs, which means answering what the MIL text does not say: the
gate order of the `ios17.lstm` operation, whether its single `bias` input is the
sum of PyTorch's `b_ih` and `b_hh`, and how closely a fp32-accumulating
implementation can track Core ML's fp16 one.

This script answers all three. It reads the weights out of the two `.mlmodelc`
bundles, reimplements both programs in NumPy, runs the compiled models on seeded
pseudo-random inputs, and reports the agreement. It then writes the fixture the
Swift test replays, so the shipping implementation is checked against captured
Core ML outputs rather than against this script.

    scripts/capture-rnnt-parity.py --out FIXTURE.json

`--model-dir` defaults to the app's Core ML model directory.

## What the numbers mean

The gate order is decided outright: only one of the four candidate blockings
reproduces the compiled decoder at all, and the rest are wrong by whole units.

Agreement within that ordering is not bit-exact and cannot be made so. Core ML's
CPU `lstm` accumulates the recurrent matrix product in fp16: feeding a zero
`h_in`, which drops the `weight_hh` term, agrees to about one fp16 ulp, while a
nonzero `h_in` disagrees by up to 0.02, and a sequential fp16 accumulation
reproduces about half of that gap (the rest is blocking order, which is not
observable from outside). A fp32 accumulation over fp16 weights is the more
accurate of the two, so the native loop uses it and this script measures the
divergence rather than trying to reproduce the error.

Measured on 32 pseudo-random tokens, which is a harsher trajectory than a real
decode: the decoder output the joint consumes agrees to 0.014 with Core ML's
state and 0.017 when the native state feeds itself, against cell values that
reach 9.2. The divergence does not compound. Whether it ever changes a token is
not decided here — the gold corpus decides that.

The joint agrees on every token it is given. Its softmax probability, which
FluidAudio carries as per-token confidence and nothing reads back, differs by up
to 0.015 for the same reason the logits do.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

DECODER_BUNDLE = "parakeet_unified_decoder.mlmodelc"
JOINT_BUNDLE = "parakeet_unified_joint_decision_single_step.mlmodelc"

BLOB_SENTINEL = 0xDEADBEEF
# Blob record dtypes, from coremltools' blob file writer.
BLOB_DTYPE_FP16 = 1

# `tensor<fp16, [A, B]> NAME = const()[... offset = tensor<uint64, []>(N)]`
CONST_RE = re.compile(
    r"tensor<(?P<dtype>fp16|fp32), \[(?P<shape>[0-9, ]+)\]> (?P<name>[A-Za-z0-9_]+) = const\(\)"
    r".*?offset = tensor<uint64, \[\]>\((?P<offset>\d+)\)",
    re.DOTALL,
)

# Weight tensor names in the two compiled programs. The Swift loader looks these
# up the same way, so a model whose export renames them fails loudly on both
# sides instead of loading the wrong matrix.
DECODER_TENSORS = {
    "embed": "module_prediction_embed_weight_to_fp16",
    "layer0_ih": "concat_1_to_fp16",
    "layer0_hh": "concat_2_to_fp16",
    "layer0_bias": "concat_0_to_fp16",
    "layer1_ih": "concat_4_to_fp16",
    "layer1_hh": "concat_5_to_fp16",
    "layer1_bias": "concat_3_to_fp16",
}
JOINT_TENSORS = {
    "enc_weight": "joint_module_enc_weight_to_fp16",
    "enc_bias": "joint_module_enc_bias_to_fp16",
    "pred_weight": "joint_module_pred_weight_to_fp16",
    "pred_bias": "joint_module_pred_bias_to_fp16",
    "out_weight": "joint_module_joint_net_2_weight_to_fp16",
    "out_bias": "joint_module_joint_net_2_bias_to_fp16",
}


class WeightFile:
    """The `weights/weight.bin` blob of one compiled ML Program."""

    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.blob = (bundle / "weights" / "weight.bin").read_bytes()
        self.consts = self._parse_mil(bundle / "model.mil")

    @staticmethod
    def _parse_mil(mil: Path) -> dict[str, tuple[tuple[int, ...], int]]:
        found: dict[str, tuple[tuple[int, ...], int]] = {}
        for match in CONST_RE.finditer(mil.read_text()):
            shape = tuple(int(part) for part in match.group("shape").split(","))
            found[match.group("name")] = (shape, int(match.group("offset")))
        return found

    def tensor(self, name: str) -> np.ndarray:
        shape, offset = self.consts[name]
        sentinel, dtype, size, data_offset = struct.unpack_from("<IIQQ", self.blob, offset)
        if sentinel != BLOB_SENTINEL:
            raise SystemExit(f"{self.bundle.name}: {name} has no blob record at {offset}")
        if dtype != BLOB_DTYPE_FP16:
            raise SystemExit(f"{self.bundle.name}: {name} is dtype {dtype}, expected fp16")
        count = int(np.prod(shape))
        if size != count * 2:
            raise SystemExit(f"{self.bundle.name}: {name} is {size} bytes, expected {count * 2}")
        flat = np.frombuffer(self.blob, dtype="<f2", count=count, offset=data_offset)
        return flat.reshape(shape).astype(np.float32)


def fp16(x: np.ndarray) -> np.ndarray:
    """Round a fp32 intermediate the way an fp16 op boundary does."""
    return x.astype(np.float16).astype(np.float32)


def packed(x: np.ndarray) -> str:
    """Base64 little-endian fp16, which every value in the fixture already is.

    Both programs cast to fp16 at their boundaries, so the outputs Core ML
    returns as fp32 round-trip exactly, and the inputs are generated fp16 so
    they do too. Storing fp32 text instead would be four times the bytes for no
    extra information.
    """
    rounded = x.astype(np.float32)
    if not np.array_equal(rounded, fp16(rounded)):
        raise SystemExit("fixture value is not representable in fp16")
    return base64.b64encode(rounded.astype("<f2").tobytes()).decode()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class PredictionNetwork:
    """The `parakeet_unified_decoder` program, in NumPy.

    `gate_order` names the row blocking of `weight_ih` / `weight_hh` / `bias`:
    MIL documents `i, f, o, g` and PyTorch exports `i, f, g, o`, and only a
    comparison against the compiled model says which survived conversion.
    """

    def __init__(self, weights: WeightFile, gate_order: str = "ifog") -> None:
        self.embed = weights.tensor(DECODER_TENSORS["embed"])
        self.layers = [
            (
                weights.tensor(DECODER_TENSORS[f"layer{index}_ih"]),
                weights.tensor(DECODER_TENSORS[f"layer{index}_hh"]),
                weights.tensor(DECODER_TENSORS[f"layer{index}_bias"]),
            )
            for index in (0, 1)
        ]
        self.gate_order = gate_order
        self.hidden = self.layers[0][2].shape[0] // 4

    def step(
        self, token: int, h: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One `(token, h, c)` to `(decoder, h_out, c_out)` step, fp32 arrays."""
        x = self.embed[token]
        h_out = np.empty_like(h)
        c_out = np.empty_like(c)
        for layer, (w_ih, w_hh, bias) in enumerate(self.layers):
            gates = fp16(w_ih @ x + w_hh @ h[layer] + bias)
            size = self.hidden
            block = {
                name: gates[index * size : (index + 1) * size]
                for index, name in enumerate(self.gate_order)
            }
            new_c = fp16(
                sigmoid(block["f"]) * c[layer] + sigmoid(block["i"]) * np.tanh(block["g"])
            )
            new_h = fp16(sigmoid(block["o"]) * np.tanh(new_c))
            h_out[layer] = new_h
            c_out[layer] = new_c
            x = new_h
        return x, h_out, c_out


class JointDecision:
    """The `parakeet_unified_joint_decision_single_step` program, in NumPy."""

    def __init__(self, weights: WeightFile) -> None:
        for field, name in JOINT_TENSORS.items():
            setattr(self, field, weights.tensor(name))

    def project_encoder(self, encoder_step: np.ndarray) -> np.ndarray:
        return fp16(self.enc_weight @ fp16(encoder_step) + self.enc_bias)

    def project_decoder(self, decoder_step: np.ndarray) -> np.ndarray:
        return fp16(self.pred_weight @ fp16(decoder_step) + self.pred_bias)

    def decide(self, enc_proj: np.ndarray, dec_proj: np.ndarray) -> tuple[int, float]:
        activated = np.maximum(fp16(enc_proj + dec_proj), 0.0)
        logits = fp16(self.out_weight @ activated + self.out_bias)
        token = int(np.argmax(logits))
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return token, float(fp16(probabilities)[token])


def load_compiled(path: Path):
    import coremltools as ct

    return ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)


def main() -> int:
    default_dir = (
        Path.home()
        / "Library/Application Support/com.parakeet.rs/models/coreml/parakeet-unified-en-0.6b"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default_dir)
    parser.add_argument("--out", type=Path, default=None, help="write the Swift test fixture here")
    parser.add_argument("--steps", type=int, default=32, help="decoder steps to compare")
    parser.add_argument(
        "--fixture-steps",
        type=int,
        default=8,
        help="steps and joint cases written to the fixture, which is checked in",
    )
    parser.add_argument("--seed", type=int, default=20264)
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=0.02,
        help="largest accepted difference in the emitted token's softmax probability",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="largest accepted absolute difference against a Core ML step",
    )
    arguments = parser.parse_args()

    decoder_bundle = arguments.model_dir / DECODER_BUNDLE
    joint_bundle = arguments.model_dir / JOINT_BUNDLE
    for bundle in (decoder_bundle, joint_bundle):
        if not bundle.is_dir():
            raise SystemExit(f"missing {bundle}")

    decoder_weights = WeightFile(decoder_bundle)
    joint_weights = WeightFile(joint_bundle)
    decoder_model = load_compiled(decoder_bundle)
    joint_model = load_compiled(joint_bundle)

    layers = 2
    hidden = decoder_weights.consts[DECODER_TENSORS["embed"]][0][1]
    vocabulary = decoder_weights.consts[DECODER_TENSORS["embed"]][0][0]
    blank = vocabulary - 1
    encoder_dim = joint_weights.consts[JOINT_TENSORS["enc_weight"]][0][1]

    rng = np.random.default_rng(arguments.seed)
    tokens = [blank] + [int(rng.integers(0, vocabulary)) for _ in range(arguments.steps - 1)]

    orders = ("ifog", "ifgo", "iofg", "igfo")
    candidates = {order: PredictionNetwork(decoder_weights, order) for order in orders}
    worst = dict.fromkeys(orders, 0.0)

    # Teacher forcing: every candidate sees the state Core ML produced, so the
    # comparison isolates one step instead of measuring accumulated drift.
    h = np.zeros((layers, hidden), dtype=np.float32)
    c = np.zeros((layers, hidden), dtype=np.float32)
    fixture_steps = []
    reference_decoder = []
    for token in tokens:
        prediction = decoder_model.predict(
            {
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h.reshape(layers, 1, hidden),
                "c_in": c.reshape(layers, 1, hidden),
            }
        )
        reference = prediction["decoder"].reshape(hidden).astype(np.float32)
        reference_h = prediction["h_out"].reshape(layers, hidden).astype(np.float32)
        reference_c = prediction["c_out"].reshape(layers, hidden).astype(np.float32)
        for order, candidate in candidates.items():
            out, out_h, out_c = candidate.step(token, h, c)
            worst[order] = max(
                worst[order],
                float(np.abs(reference - out).max()),
                float(np.abs(reference_h - out_h).max()),
                float(np.abs(reference_c - out_c).max()),
            )
        fixture_steps.append(
            {
                "token": token,
                "h_in": packed(h.reshape(-1)),
                "c_in": packed(c.reshape(-1)),
                "decoder": packed(reference),
                "h_out": packed(reference_h.reshape(-1)),
                "c_out": packed(reference_c.reshape(-1)),
            }
        )
        reference_decoder.append(reference)
        h, c = reference_h, reference_c

    print("prediction network, worst absolute difference over one step:")
    for order in sorted(orders, key=lambda key: worst[key]):
        print(f"  gates={order}  worst={worst[order]:.6f}")
    best = min(orders, key=lambda key: worst[key])
    runner_up = sorted(worst.values())[1]
    if best != "ifog" or worst[best] > arguments.tolerance or runner_up < 10 * worst[best]:
        print(
            f"gate order is not decided: best {best} at {worst[best]:.6f}, "
            f"runner-up at {runner_up:.6f}",
            file=sys.stderr,
        )
        return 1
    print(f"gate order {best}, within {worst[best]:.6f} of the compiled decoder")

    # Free running: the native state feeds itself, which is what the decode loop
    # does, so this is the divergence that can actually reach a token decision.
    native = candidates[best]
    native_h = np.zeros((layers, hidden), dtype=np.float32)
    native_c = np.zeros((layers, hidden), dtype=np.float32)
    divergence = 0.0
    for step, token in enumerate(tokens):
        out, native_h, native_c = native.step(token, native_h, native_c)
        divergence = max(divergence, float(np.abs(reference_decoder[step] - out).max()))
    print(f"free-running divergence after {len(tokens)} steps: {divergence:.6f}")
    if divergence > arguments.tolerance:
        print(f"divergence exceeds {arguments.tolerance}", file=sys.stderr)
        return 1

    joint = JointDecision(joint_weights)
    mismatches = 0
    worst_probability = 0.0
    emitting = 0
    decoder_output = reference_decoder[0]
    joint_cases = []
    for index in range(arguments.steps):
        # Half the cases are noise, which the joint answers with blank however
        # it is scaled — the blank logit dominates everything an encoder never
        # produces. The other half are built to emit: for a target token the
        # activation that separates it from blank is
        # `relu(W_out[token] - W_out[blank])`, and the encoder projection is
        # linear, so least squares gives an encoder step that lands there. The
        # captured answer is whatever the compiled model then returns, which is
        # what makes these references rather than expectations.
        if index % 2 == 0:
            encoder_step = fp16(rng.normal(0.0, 1.0, size=encoder_dim).astype(np.float32))
        else:
            target = int(rng.integers(0, vocabulary - 1))
            wanted = np.maximum(joint.out_weight[target] - joint.out_weight[blank], 0.0)
            wanted = wanted / max(np.linalg.norm(wanted), 1e-6) * 8.0
            residual = wanted - joint.enc_bias - joint.project_decoder(decoder_output)
            encoder_step = fp16(
                np.linalg.lstsq(joint.enc_weight, residual, rcond=None)[0].astype(np.float32)
            )
        prediction = joint_model.predict(
            {
                "encoder_step": encoder_step.reshape(1, encoder_dim, 1),
                "decoder_step": decoder_output.reshape(1, hidden, 1),
            }
        )
        reference_token = int(np.asarray(prediction["token_id"]).reshape(-1)[0])
        reference_probability = float(np.asarray(prediction["token_prob"]).reshape(-1)[0])
        token, probability = joint.decide(
            joint.project_encoder(encoder_step), joint.project_decoder(decoder_output)
        )
        mismatches += int(token != reference_token)
        emitting += int(reference_token != blank)
        worst_probability = max(worst_probability, abs(probability - reference_probability))
        joint_cases.append(
            {
                "encoder_step": packed(encoder_step),
                "decoder_step": packed(decoder_output),
                "token_id": reference_token,
                "token_prob": reference_probability,
            }
        )
        decoder_output = reference_decoder[(index + 1) % len(reference_decoder)]

    print(
        f"joint decision: {mismatches} argmax mismatches over {arguments.steps} cases "
        f"({emitting} of them emitting), "
        f"worst probability difference {worst_probability:.3e}"
    )
    if mismatches:
        return 1
    if worst_probability > arguments.probability_tolerance:
        print(
            f"probability differs by {worst_probability:.3e}, over "
            f"{arguments.probability_tolerance}",
            file=sys.stderr,
        )
        return 1
    if emitting < arguments.fixture_steps // 2:
        print(
            f"only {emitting} cases emit a token; a corpus of blanks would not exercise "
            "the argmax or the probability",
            file=sys.stderr,
        )
        return 1

    # Frames for the loop-level test: strong enough to emit from the state a
    # freshly reset decoder is in, whatever the trajectory does to it after the
    # first token. The joint case corpus above cannot serve here because each of
    # its constructed steps was aimed at the decoder state of its own step.
    loop_frames = []
    start_projection = joint.project_decoder(reference_decoder[0])
    for _ in range(arguments.fixture_steps):
        target = int(rng.integers(0, vocabulary - 1))
        wanted = np.maximum(joint.out_weight[target] - joint.out_weight[blank], 0.0)
        wanted = wanted / max(np.linalg.norm(wanted), 1e-6) * 40.0
        residual = wanted - joint.enc_bias - start_projection
        loop_frames.append(
            packed(
                fp16(np.linalg.lstsq(joint.enc_weight, residual, rcond=None)[0].astype(np.float32))
            )
        )

    if arguments.out:
        fixture = {
            "note": (
                "Captured Core ML outputs for the Parakeet Unified prediction network and "
                "joint decision head. Regenerate with scripts/capture-rnnt-parity.py."
            ),
            "decoder_weights_sha256": hashlib.sha256(decoder_weights.blob).hexdigest(),
            "joint_weights_sha256": hashlib.sha256(joint_weights.blob).hexdigest(),
            "gate_order": best,
            "layers": layers,
            "hidden": hidden,
            "encoder_dim": encoder_dim,
            "vocabulary": vocabulary,
            "step_tolerance": arguments.tolerance,
            "probability_tolerance": arguments.probability_tolerance,
            "decoder_steps": fixture_steps[: arguments.fixture_steps],
            "joint_cases": joint_cases[: arguments.fixture_steps],
            "loop_frames": loop_frames,
        }
        arguments.out.write_text(json.dumps(fixture))
        print(f"wrote {arguments.out} ({arguments.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
