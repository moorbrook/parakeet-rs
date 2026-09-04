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
        // A local override, not a vendored copy. FluidAudio hardcodes the
        // offline encoder window at 15 s, which is the whole of what bucketed
        // short-window encoders need to change (kata fgzt), so
        // `scripts/build-coreml-worker.sh` clones
        // https://github.com/FluidInference/FluidAudio.git at revision
        // 00a9aa771900ea09c485659663be31019e293e47 into `.fluidaudio-local` and
        // applies `patches/fluidaudio-offline-window.patch`. Run that script
        // before building this package. Restore the plain SCM dependency and
        // delete both the patch and the clone if the change lands upstream.
        .package(name: "FluidAudio", path: ".fluidaudio-local"),
    ],
    targets: [
        .executableTarget(
            name: "ParakeetCoreMLWorker",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/ParakeetCoreMLWorker"
        ),
        .testTarget(
            name: "ParakeetCoreMLWorkerTests",
            dependencies: ["ParakeetCoreMLWorker"],
            path: "Tests/ParakeetCoreMLWorkerTests"
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
