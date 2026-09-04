// Compute-plan reporter: prints, for every operation of a compiled Core ML
// model, the device Core ML prefers for it and the devices it could run on.
//
// This is a diagnostic used to answer whether the encoder really has no
// CPU/GPU fallback under `cpuAndNeuralEngine`. It is not part of the
// dictation path and links no FluidAudio code.
import CoreML
import Foundation

private struct Options {
    var modelURLs: [URL] = []
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var format: String = "markdown"
}

private func computeUnitsName(_ units: MLComputeUnits) -> String {
    switch units {
    case .all: "all"
    case .cpuAndGPU: "cpu-and-gpu"
    case .cpuAndNeuralEngine: "cpu-and-neural-engine"
    case .cpuOnly: "cpu-only"
    @unknown default: "unknown"
    }
}

private func parseComputeUnits(_ value: String) -> MLComputeUnits? {
    switch value {
    case "all": .all
    case "cpu-and-gpu": .cpuAndGPU
    case "cpu-and-neural-engine": .cpuAndNeuralEngine
    case "cpu-only": .cpuOnly
    default: nil
    }
}

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("parakeet-compute-plan: \(message)\n".utf8))
    exit(1)
}

private func parseOptions() -> Options {
    var options = Options()
    let arguments = Array(CommandLine.arguments.dropFirst())
    var index = 0
    while index < arguments.count {
        switch arguments[index] {
        case "--compute-units":
            index += 1
            guard index < arguments.count, let units = parseComputeUnits(arguments[index]) else {
                fail("--compute-units needs all|cpu-and-gpu|cpu-and-neural-engine|cpu-only")
            }
            options.computeUnits = units
        case "--format":
            index += 1
            guard index < arguments.count, ["markdown", "csv"].contains(arguments[index]) else {
                fail("--format needs markdown or csv")
            }
            options.format = arguments[index]
        case "-h", "--help":
            FileHandle.standardError.write(
                Data(
                    "usage: parakeet-compute-plan [--compute-units NAME] [--format markdown|csv] MODEL.mlmodelc...\n"
                        .utf8))
            exit(0)
        default:
            options.modelURLs.append(
                URL(fileURLWithPath: arguments[index], isDirectory: true))
        }
        index += 1
    }
    guard !options.modelURLs.isEmpty else { fail("no model paths given") }
    return options
}

@available(macOS 14.4, *)
private func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: "CPU"
    case .gpu: "GPU"
    case .neuralEngine: "ANE"
    @unknown default: "unknown"
    }
}

private struct OperationRecord {
    let path: String
    let operatorName: String
    let outputName: String
    let preferred: String
    let supported: String
    let weight: Double
}

@available(macOS 14.4, *)
private func walk(
    block: MLModelStructure.Program.Block,
    plan: MLComputePlan,
    prefix: String,
    into records: inout [OperationRecord],
    constCost: inout Double
) {
    for (index, operation) in block.operations.enumerated() {
        let path = prefix.isEmpty ? "\(index)" : "\(prefix).\(index)"
        if operation.operatorName == "const" {
            constCost += plan.estimatedCost(of: operation)?.weight ?? 0
        }
        // `const` is literal data, not scheduled work, and there are ~2000 of
        // them; its estimated cost is summed separately by the caller.
        if operation.operatorName != "const" {
            let usage = plan.deviceUsage(for: operation)
            records.append(
                OperationRecord(
                    path: path,
                    operatorName: operation.operatorName,
                    outputName: operation.outputs.first?.name ?? "",
                    preferred: usage.map { deviceName($0.preferred) } ?? "n/a",
                    supported: (usage?.supported ?? []).map(deviceName).sorted().joined(
                        separator: "+"),
                    weight: plan.estimatedCost(of: operation)?.weight ?? 0
                ))
        }
        for (blockIndex, nested) in operation.blocks.enumerated() {
            walk(
                block: nested, plan: plan, prefix: "\(path).b\(blockIndex)", into: &records,
                constCost: &constCost)
        }
    }
}

@available(macOS 14.4, *)
private func records(for plan: MLComputePlan) -> ([OperationRecord], Double) {
    var result: [OperationRecord] = []
    var constCost = 0.0
    switch plan.modelStructure {
    case .program(let program):
        for name in program.functions.keys.sorted() {
            guard let function = program.functions[name] else { continue }
            walk(
                block: function.block, plan: plan, prefix: name, into: &result,
                constCost: &constCost)
        }
    case .neuralNetwork:
        fail("model is a NeuralNetwork, not an ML Program; no per-op plan available")
    case .pipeline:
        fail("model is a pipeline; unwrap its sub-models first")
    case .unsupported:
        fail("Core ML reports this model structure as unsupported")
    @unknown default:
        fail("unrecognized model structure")
    }
    return (result, constCost)
}

@available(macOS 14.4, *)
private func runReport() async {
    let options = parseOptions()


    for url in options.modelURLs {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = options.computeUnits
        let plan: MLComputePlan
        do {
            plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        } catch {
            fail("could not load a compute plan for \(url.lastPathComponent): \(error)")
        }
        let (operations, constCost) = records(for: plan)

        var byDevice: [String: Int] = [:]
        var costByDevice: [String: Double] = [:]
        var byOperator: [String: [String: Int]] = [:]
        for record in operations {
            byDevice[record.preferred, default: 0] += 1
            costByDevice[record.preferred, default: 0] += record.weight
            byOperator[record.operatorName, default: [:]][record.preferred, default: 0] += 1
        }

        if options.format == "csv" {
            if url == options.modelURLs.first {
                print("model,op_index,operator,output,preferred,supported,estimated_cost")
            }
            for record in operations {
                print(
                    "\(url.lastPathComponent),\(record.path),\(record.operatorName),\(record.outputName),\(record.preferred),\(record.supported),\(record.weight)"
                )
            }
            continue
        }

        print("## \(url.lastPathComponent)")
        print("")
        print("compute units requested: `\(computeUnitsName(options.computeUnits))`")
        print("non-const operations: \(operations.count)")
        print("")
        let totalCost = costByDevice.values.reduce(0, +) + constCost
        print("| preferred device | operations | estimated cost weight |")
        print("|---|---:|---:|")
        for key in byDevice.keys.sorted() {
            let share = (costByDevice[key] ?? 0) * 100
            print("| \(key) | \(byDevice[key] ?? 0) | \(String(format: "%.2f", share))% |")
        }
        print("| (const, not scheduled) | — | \(String(format: "%.2f", constCost * 100))% |")
        print("| **total** | \(operations.count) | \(String(format: "%.2f", totalCost * 100))% |")
        print("")
        print("| operator | ANE | CPU | GPU | other |")
        print("|---|---:|---:|---:|---:|")
        for name in byOperator.keys.sorted() {
            let counts = byOperator[name] ?? [:]
            let known = ["ANE", "CPU", "GPU"]
            let other = counts.filter { !known.contains($0.key) }.values.reduce(0, +)
            print(
                "| `\(name)` | \(counts["ANE"] ?? 0) | \(counts["CPU"] ?? 0) | \(counts["GPU"] ?? 0) | \(other) |"
            )
        }
        print("")
        let unplaced = operations.filter { $0.preferred == "n/a" }
        let notANE = operations.filter { $0.preferred != "ANE" && $0.preferred != "n/a" }
        print(
            "operations with no device assignment (weight decompression, folded into the consumer): \(unplaced.count)"
        )
        print("")
        print("non-ANE scheduled operations: \(notANE.count)")
        if !notANE.isEmpty {
            print("")
            print("| op index | operator | output | preferred | supported | estimated cost |")
            print("|---|---|---|---|---|---:|")
            for record in notANE {
                print(
                    "| \(record.path) | `\(record.operatorName)` | `\(record.outputName)` | \(record.preferred) | \(record.supported) | \(String(format: "%.5f", record.weight)) |"
                )
            }
        }
        print("")
    }
}

if #available(macOS 14.4, *) {
    await runReport()
} else {
    fail("MLComputePlan requires macOS 14.4 or newer")
}
