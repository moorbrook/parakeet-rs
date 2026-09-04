// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ParakeetCoreMLWorker",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(
            name: "parakeet-coreml-worker",
            targets: ["ParakeetCoreMLWorker"]
        ),
        .executable(
            name: "parakeet-compute-plan",
            targets: ["ParakeetComputePlan"]
        ),
        .executable(
            name: "parakeet-encoder-probe",
            targets: ["ParakeetEncoderProbe"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "00a9aa771900ea09c485659663be31019e293e47"
        ),
    ],
    targets: [
        .executableTarget(
            name: "ParakeetCoreMLWorker",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/ParakeetCoreMLWorker"
        ),
        // Diagnostic only: reports the Core ML compute plan per operation.
        // Deliberately does not depend on FluidAudio.
        .executableTarget(
            name: "ParakeetComputePlan",
            path: "Sources/ParakeetComputePlan"
        ),
        // Diagnostic only: reports an encoder's compiled shapes and its cost on
        // zero inputs. Deliberately does not depend on FluidAudio.
        .executableTarget(
            name: "ParakeetEncoderProbe",
            path: "Sources/ParakeetEncoderProbe"
        ),
    ]
)
