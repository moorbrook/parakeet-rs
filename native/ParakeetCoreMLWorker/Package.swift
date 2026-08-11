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
    ]
)
