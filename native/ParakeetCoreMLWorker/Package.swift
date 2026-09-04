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
        // A local override, not a vendored copy. Two things in FluidAudio have
        // to change here: the offline encoder window is hardcoded at 15 s,
        // which bucketed short-window encoders need parameterized (kata fgzt),
        // and the greedy RNNT decode loop is fixed to its two CoreML models,
        // which `NativeRnntDecoder` replaces (kata 2564). So
        // `scripts/build-coreml-worker.sh` clones
        // https://github.com/FluidInference/FluidAudio.git at revision
        // 00a9aa771900ea09c485659663be31019e293e47 into `.fluidaudio-local` and
        // applies `patches/fluidaudio.patch`. Run that script before building
        // this package. Restore the plain SCM dependency and delete both the
        // patch and the clone if the changes land upstream.
        .package(name: "FluidAudio", path: ".fluidaudio-local"),
    ],
    targets: [
        .executableTarget(
            name: "ParakeetCoreMLWorker",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
                "ParakeetRnntKernels",
            ],
            path: "Sources/ParakeetCoreMLWorker"
        ),
        // The two inner loops of the native RNNT decode loop, in C because
        // Swift compiles a fp16-to-fp32 SIMD conversion to an outlined runtime
        // call. See the header.
        .target(
            name: "ParakeetRnntKernels",
            path: "Sources/ParakeetRnntKernels"
        ),
        .testTarget(
            name: "ParakeetCoreMLWorkerTests",
            dependencies: ["ParakeetCoreMLWorker"],
            path: "Tests/ParakeetCoreMLWorkerTests",
            resources: [.copy("Fixtures/rnnt-parity.json")]
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
