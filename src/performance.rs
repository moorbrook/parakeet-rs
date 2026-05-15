//! Apple-Silicon CPU topology helpers. Tells the ASR how many threads to
//! launch (= performance-core count), so the encoder doesn't get scheduled
//! onto the efficiency cores under load.

use std::ffi::CString;

/// `sysctlbyname("hw.perflevel0.logicalcpu")` returns the P-core count on
/// Apple Silicon (M1+). Falls back to half of total logicals if the sysctl
/// is missing — non-Apple-Silicon Macs, hypothetically.
pub fn performance_core_count() -> i32 {
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    let name = CString::new("hw.perflevel0.logicalcpu").unwrap();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        value
    } else {
        (num_cpus_total() / 2).max(2) as i32
    }
}

fn num_cpus_total() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
