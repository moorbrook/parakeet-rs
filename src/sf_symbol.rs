//! Native SF Symbols rendering for the menu-bar status icon.
//!
//! macOS exposes the SF Symbols library through `NSImage(systemSymbolName:)`.
//! For our case (`NSStatusItem.button.image = ...`) we hand back the raw
//! `NSImage` directly with `setTemplate(true)` so the menu bar recolours it
//! for light/dark mode and selection state automatically.

use objc2::rc::Retained;
use objc2_app_kit::NSImage;
use objc2_foundation::{NSSize, NSString};

/// Resolve an SF Symbol by name and configure it as a template image at the
/// given point size. Returns `None` if the symbol doesn't exist (e.g. older
/// macOS than the symbol was introduced in).
pub fn load(name: &str, point_size: f64) -> Option<Retained<NSImage>> {
    unsafe {
        let name_ns = NSString::from_str(name);
        let image = NSImage::imageWithSystemSymbolName_accessibilityDescription(&name_ns, None)?;
        image.setSize(NSSize {
            width: point_size,
            height: point_size,
        });
        // Template images get recoloured by the system per appearance.
        image.setTemplate(true);
        Some(image)
    }
}
