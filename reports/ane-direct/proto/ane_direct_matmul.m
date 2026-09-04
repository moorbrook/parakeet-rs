// ane_direct_matmul.m — minimal proof that a hand-authored MIL program can be
// compiled and evaluated on the Apple Neural Engine through the private
// AppleNeuralEngine.framework Objective-C interface, with no Core ML in the path.
//
// The program is a single fp16 [64,64] matrix multiply expressed as a 1x1
// convolution (the ANE's preferred matmul formulation), run over a [1,64,1,16]
// activation. The result is checked against an fp32 CPU reference.
//
// Research spike. Private API, unsupported, version-fragile. Not production code.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
//     -o ane_direct_matmul ane_direct_matmul.m
// Run:
//   ./ane_direct_matmul            # one compile + one evaluate, no timing loop

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <math.h>

#define CH   64      // input and output channels
#define SEQ  16      // sequence length; the minimum decode bucket
#define SURF_BYTES 65536  // >= the ~49 KB minimum IOSurface the engine demands at eval

// A weight blob: 128-byte container header, fp16 payload. The MIL const() offset
// of 64 points at the chunk header inside this layout, not at the payload.
static NSData *make_blobfile(const float *src, int count) {
    size_t payload = (size_t)count * sizeof(uint16_t);
    size_t total = 128 + payload;
    uint8_t *b = (uint8_t *)calloc(total, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = (uint32_t)payload;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *w = (_Float16 *)(b + 128);
    for (int i = 0; i < count; i++) w[i] = (_Float16)src[i];
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth:            @(bytes),
        (id)kIOSurfaceHeight:           @1,
        (id)kIOSurfaceBytesPerElement:  @1,
        (id)kIOSurfaceBytesPerRow:      @(bytes),
        (id)kIOSurfaceAllocSize:        @(bytes),
        (id)kIOSurfacePixelFormat:      @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static NSString *build_mil(NSString *weight_path) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n"
                    "[buildInfo = dict<string, string>({"
                    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                    "{\"coremlc-version\", \"3505.4.1\"}, "
                    "{\"coremltools-component-milinternal\", \"\"}, "
                    "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", CH, SEQ];
    [m appendString:@"        string c_pt = const()[name=string(\"c_pt\"), val=string(\"valid\")];\n"
                    "        tensor<int32, [2]> c_st = const()[name=string(\"c_st\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        tensor<int32, [4]> c_pd = const()[name=string(\"c_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                    "        tensor<int32, [2]> c_dl = const()[name=string(\"c_dl\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        int32 c_gr = const()[name=string(\"c_gr\"), val=int32(1)];\n"];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> w = const()[name=string(\"w\"), "
                    "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n",
                    CH, CH, CH, CH, weight_path];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv("
                    "dilations=c_dl, groups=c_gr, pad=c_pd, pad_type=c_pt, strides=c_st, "
                    "weight=w, x=x)[name=string(\"out\")];\n", CH, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

int main(void) { @autoreleasepool {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
                     "AppleNeuralEngine", RTLD_LAZY);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
    Class Req  = NSClassFromString(@"_ANERequest");
    Class Surf = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!Desc || !IMM || !Req || !Surf) { fprintf(stderr, "private classes missing\n"); return 1; }

    // Deterministic small values so the fp16 product stays well inside range.
    static float W[CH * CH], X[CH * SEQ];
    for (int i = 0; i < CH * CH; i++)  W[i] = ((i * 37 % 17) - 8) / 32.0f;
    for (int i = 0; i < CH * SEQ; i++) X[i] = ((i * 53 % 13) - 6) / 16.0f;

    NSString *wpath = @"@model_path/weights/w.bin";
    NSString *mil = build_mil(wpath);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *wdict = @{ wpath: @{ @"offset": @0, @"data": make_blobfile(W, CH * CH) } };

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        Desc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!desc || !model) { fprintf(stderr, "descriptor/model creation failed\n"); return 1; }

    // The compiler service reads the program and its weights from a directory
    // named by the model's own content hash, so stage them there first.
    NSString *hexId = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
  withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdict[wpath][@"data"] writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/w.bin"]
                            atomically:YES];
    printf("staged program at %s\n", hexId.UTF8String);

    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &err);
    printf("compile: %s\n", ok ? "ok" : err.description.UTF8String);
    if (!ok) return 2;

    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &err);
    printf("load: %s\n", ok ? "ok" : err.description.UTF8String);
    if (!ok) return 3;

    IOSurfaceRef in = make_surface(SURF_BYTES), out = make_surface(SURF_BYTES);
    IOSurfaceLock(in, 0, NULL);
    _Float16 *ip = (_Float16 *)IOSurfaceGetBaseAddress(in);
    for (int i = 0; i < CH * SEQ; i++) ip[i] = (_Float16)X[i];
    IOSurfaceUnlock(in, 0, NULL);

    id inObj  = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        Surf, @selector(objectWithIOSurface:), in);
    id outObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        Surf, @selector(objectWithIOSurface:), out);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        Req, @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                       weightsBuffer:perfStats:procedureIndex:),
        @[inObj], @[@0], @[outObj], @[@0], nil, nil, @0);

    ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &err);
    printf("evaluate: %s\n", ok ? "ok" : err.description.UTF8String);
    if (!ok) return 4;

    // The engine's internal activation layout pads the last axis; rather than
    // assume one, score the output under several candidate strides in the single
    // evaluation we are allowed, and report which one reproduces the reference.
    IOSurfaceLock(out, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *op = (const _Float16 *)IOSurfaceGetBaseAddress(out);
    static double ref[CH * SEQ];
    for (int o = 0; o < CH; o++)
        for (int s = 0; s < SEQ; s++) {
            double a = 0.0;
            for (int i = 0; i < CH; i++) a += (double)W[o * CH + i] * (double)X[i * SEQ + s];
            ref[o * SEQ + s] = a;
        }
    const int strides[] = { SEQ, 32, 64, 128 };
    double best = 1e9; int best_stride = 0; int best_transposed = 0;
    for (int k = 0; k < 4; k++) {
        for (int tr = 0; tr < 2; tr++) {
            double worst_k = 0.0;
            for (int o = 0; o < CH; o++)
                for (int s = 0; s < SEQ; s++) {
                    double got = tr ? (double)op[s * strides[k] + o]
                                    : (double)op[o * strides[k] + s];
                    double r = ref[o * SEQ + s];
                    double denom = fabs(r) > 1.0 ? fabs(r) : 1.0;
                    double rel = fabs(got - r) / denom;
                    if (rel > worst_k) worst_k = rel;
                }
            printf("  stride=%3d %-12s max rel err %.5f\n",
                   strides[k], tr ? "transposed" : "channel-major", worst_k);
            if (worst_k < best) { best = worst_k; best_stride = strides[k]; best_transposed = tr; }
        }
    }
    IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, NULL);
    double worst = best;
    printf("best: stride=%d %s, max relative error vs fp32 CPU reference %.5f\n",
           best_stride, best_transposed ? "transposed" : "channel-major", best);
    printf("%s\n", worst < 0.01 ? "PARITY OK" : "PARITY FAIL");

    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model, @selector(unloadWithQoS:error:), 21, &err);
    [fm removeItemAtPath:tmpDir error:nil];
    CFRelease(in); CFRelease(out);
    return worst < 0.01 ? 0 : 5;
} }
