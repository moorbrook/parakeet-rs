import Accelerate
import Foundation
import ParakeetRnntKernels

/// The weights of a compiled ML Program, read straight out of its bundle.
///
/// A `.mlmodelc` directory carries its constants in `weights/weight.bin` and
/// names them in `model.mil`, which is plain text. Reading them here is what
/// lets the RNNT decode loop run natively instead of through a Core ML
/// dispatch per step: the same numbers, without the driver round trip.
///
/// Nothing about this is a general MIL reader. It resolves named fp16
/// constants and validates every one against the blob record that precedes it,
/// so a model whose export renames or reshapes a tensor fails here rather than
/// loading the wrong matrix.
struct MilWeightBlob {
    /// A blob record: 32 bytes of metadata, 64-byte aligned, ahead of the data.
    private static let sentinel: UInt32 = 0xDEAD_BEEF
    private static let fp16DataType: UInt32 = 1

    private let blob: Data
    private let constants: [String: (shape: [Int], offset: Int)]

    init(bundle: URL) throws {
        blob = try Data(
            contentsOf: bundle.appendingPathComponent("weights/weight.bin"),
            options: .mappedIfSafe
        )
        let program = try String(
            contentsOf: bundle.appendingPathComponent("model.mil"), encoding: .utf8)
        constants = try Self.parse(program)
    }

    /// `tensor<fp16, [1025, 640]> NAME = const()[... offset = tensor<uint64, []>(64)]`
    private static func parse(_ program: String) throws -> [String: (shape: [Int], offset: Int)] {
        // `[^;]*?` cannot cross a statement boundary, so a const can only ever
        // be paired with its own BLOBFILE offset.
        let expression = try NSRegularExpression(
            pattern:
                #"tensor<fp16, \[([0-9, ]+)\]> ([A-Za-z0-9_]+) = const\(\)[^;]*?"#
                + #"offset = tensor<uint64, \[\]>\((\d+)\)"#
        )
        var found: [String: (shape: [Int], offset: Int)] = [:]
        let range = NSRange(program.startIndex..<program.endIndex, in: program)
        for match in expression.matches(in: program, range: range) {
            guard let shapeRange = Range(match.range(at: 1), in: program),
                let nameRange = Range(match.range(at: 2), in: program),
                let offsetRange = Range(match.range(at: 3), in: program),
                let offset = Int(program[offsetRange])
            else { continue }
            let shape = program[shapeRange]
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            found[String(program[nameRange])] = (shape, offset)
        }
        return found
    }

    enum LoadError: LocalizedError {
        case missingConstant(String)
        case shapeMismatch(String, [Int], [Int])
        case corruptRecord(String)

        var errorDescription: String? {
            switch self {
            case .missingConstant(let name):
                "compiled model has no fp16 constant named \(name)"
            case .shapeMismatch(let name, let found, let wanted):
                "\(name) has shape \(found), expected \(wanted)"
            case .corruptRecord(let name):
                "\(name) does not point at an fp16 blob record of the declared size"
            }
        }
    }

    /// The declared shape of a named fp16 constant.
    func shape(_ name: String) throws -> [Int] {
        guard let entry = constants[name] else { throw LoadError.missingConstant(name) }
        return entry.shape
    }

    /// The named constant as fp16 values, checked against `shape`.
    func fp16(_ name: String, shape wanted: [Int]) throws -> [Float16] {
        guard let entry = constants[name] else { throw LoadError.missingConstant(name) }
        guard entry.shape == wanted else {
            throw LoadError.shapeMismatch(name, entry.shape, wanted)
        }
        let count = wanted.reduce(1, *)
        return try blob.withUnsafeBytes { raw -> [Float16] in
            guard entry.offset >= 0, entry.offset + 32 <= raw.count else {
                throw LoadError.corruptRecord(name)
            }
            let record = raw.baseAddress! + entry.offset
            let sentinel = record.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
            let dataType = record.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
            let byteCount = Int(record.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
            let dataOffset = Int(record.loadUnaligned(fromByteOffset: 16, as: UInt64.self))
            guard sentinel == Self.sentinel, dataType == Self.fp16DataType,
                byteCount == count * MemoryLayout<Float16>.size,
                dataOffset >= 0, dataOffset + byteCount <= raw.count
            else { throw LoadError.corruptRecord(name) }
            return [Float16](
                unsafeUninitializedCapacity: count,
                initializingWith: { buffer, initialized in
                    memcpy(buffer.baseAddress!, raw.baseAddress! + dataOffset, byteCount)
                    initialized = count
                })
        }
    }

    /// The named constant widened to fp32, for the one projection that goes
    /// through BLAS rather than the fp16 row kernel.
    func fp32(_ name: String, shape: [Int]) throws -> [Float] {
        try fp16(name, shape: shape).map(Float.init)
    }
}

/// A row-major fp16 matrix the row kernel reads directly.
///
/// Weights stay fp16 because the decode loop is bound by how fast their bytes
/// arrive, not by arithmetic: one prediction-network step reads 13.1 MB of
/// them, and widening to fp32 at rest would double that for no accuracy the
/// exported model has. Products accumulate in fp32.
final class Float16Matrix {
    let rows: Int
    let columns: Int
    private let storage: UnsafeMutablePointer<Float16>

    /// Rows are 64-byte aligned so a row load never straddles a cache line
    /// boundary it did not have to.
    init(rows: Int, columns: Int, values: [Float16]) {
        precondition(values.count == rows * columns)
        self.rows = rows
        self.columns = columns
        storage = UnsafeMutableRawPointer.allocate(
            byteCount: values.count * MemoryLayout<Float16>.size,
            alignment: 64
        ).bindMemory(to: Float16.self, capacity: values.count)
        values.withUnsafeBufferPointer { source in
            storage.update(from: source.baseAddress!, count: values.count)
        }
    }

    /// Two matrices with the same row count, interleaved row by row.
    ///
    /// The LSTM gate pre-activation is `W_ih · x + W_hh · h + b`, and the two
    /// products share an output row. Storing row `r` of both matrices back to
    /// back turns the pair into one sequential read of `2 * columns` values
    /// against a concatenated `[x, h]` vector.
    convenience init(interleavingRowsOf first: [Float16], _ second: [Float16], rows: Int) {
        let columns = first.count / rows
        precondition(first.count == rows * columns && second.count == rows * columns)
        var packed = [Float16](repeating: 0, count: 2 * rows * columns)
        for row in 0..<rows {
            let destination = row * 2 * columns
            let source = row * columns
            packed.replaceSubrange(
                destination..<(destination + columns),
                with: first[source..<(source + columns)])
            packed.replaceSubrange(
                (destination + columns)..<(destination + 2 * columns),
                with: second[source..<(source + columns)])
        }
        self.init(rows: rows, columns: 2 * columns, values: packed)
    }

    deinit { storage.deallocate() }

    /// `out[r] = fp16(bias[r] + row(r) · vector)`, over `rowRange`.
    ///
    /// The fp16 rounding of the result is not decoration: every tensor in the
    /// compiled program is fp16 at an operation boundary, so a native step that
    /// kept fp32 intermediates would follow a different trajectory than the
    /// model was exported to run.
    func multiply(
        vector: UnsafePointer<Float>,
        bias: UnsafePointer<Float>,
        into out: UnsafeMutablePointer<Float>,
        rowRange: Range<Int>
    ) {
        UnsafeRawPointer(storage).withMemoryRebound(
            to: UInt16.self, capacity: rows * columns
        ) { weights in
            parakeet_rnnt_matvec(
                weights, vector, bias, out, columns, rowRange.lowerBound, rowRange.upperBound)
        }
    }

    /// The same product over every row, split across `chunks` concurrent slices.
    ///
    /// Row slices are independent and each accumulates its own dot products, so
    /// the result does not depend on how the rows were divided.
    func multiply(
        vector: UnsafePointer<Float>,
        bias: UnsafePointer<Float>,
        into out: UnsafeMutablePointer<Float>,
        chunks: Int
    ) {
        guard chunks > 1, rows >= chunks else {
            multiply(vector: vector, bias: bias, into: out, rowRange: 0..<rows)
            return
        }
        let stride = (rows + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let start = chunk * stride
            guard start < rows else { return }
            multiply(
                vector: vector, bias: bias, into: out,
                rowRange: start..<min(start + stride, rows))
        }
    }
}

/// Round a buffer to fp16 and back, which is what a cast to an fp16 tensor does.
func roundToFloat16(_ values: UnsafeMutablePointer<Float>, count: Int) {
    parakeet_rnnt_round_fp16(values, count)
}
