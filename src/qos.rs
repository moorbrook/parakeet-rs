//! macOS thread Quality-of-Service.
//!
//! Apple Silicon schedulers honour QoS classes when deciding between
//! performance and efficiency cores. `USER_INTERACTIVE` keeps our capture and
//! recognition threads off the E-cores, which is what ds4 calls out as one of
//! the "free" wins on Apple hardware.

#[allow(non_camel_case_types)]
#[repr(C)]
enum QosClass {
    UserInteractive = 0x21,
}

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(
        qos_class: libc::c_int,
        relative_priority: libc::c_int,
    ) -> libc::c_int;
}

/// Pin the calling thread to performance cores by raising its QoS class.
pub fn set_user_interactive() {
    let rc =
        unsafe { pthread_set_qos_class_self_np(QosClass::UserInteractive as libc::c_int, 0) };
    if rc != 0 {
        log::warn!("pthread_set_qos_class_self_np returned {rc}");
    }
}
