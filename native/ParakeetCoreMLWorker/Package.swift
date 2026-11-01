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
        // The moorbrook fork, branch `parakeet-rs-window`: upstream
        // FluidInference/FluidAudio at 00a9aa771900ea09c485659663be31019e293e47
        // plus one commit with the two changes this worker needs. The offline
        // encoder window is a parameter instead of a hardcoded 15 s, for the
        // short-window encoder buckets (kata fgzt), and the greedy RNNT decode
        // loop is a protocol with a factory, which `NativeRnntDecoder`
        // implements (kata 2564). Pinned by revision; see ADR-0038.
        .package(
            url: "https://github.com/moorbrook/FluidAudio.git",
            revision: "94702495056e0fd0ed61c16af31d42d20fccd7b7"
        ),
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
