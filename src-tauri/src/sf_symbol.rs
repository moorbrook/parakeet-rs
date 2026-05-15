//! Native SF Symbols rendering for the menu-bar tray icon.
//!
//! macOS exposes the SF Symbols library through `NSImage(systemSymbolName:)`.
//! We render the symbol into an `NSBitmapImageRep`, ask AppKit for the PNG
//! representation, and hand those bytes to Tauri's `Image::from_bytes`. The
//! resulting `Image` is treated by macOS as a template image (because
//! `NSImage` flags symbol images as templates by default), so the menu bar
//! recolours it for light/dark mode and the "selected" state automatically.

use objc2::ClassType;
use objc2::rc::Retained;
use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
use objc2_foundation::{NSDictionary, NSPoint, NSRect, NSSize, NSString};
use tauri::image::Image;

/// Render an SF Symbol to PNG bytes at the given point size, returning a
/// Tauri `Image`. Falls back to `None` if AppKit cannot find the symbol —
/// the caller should always check.
pub fn load(name: &str, point_size: f64) -> Option<Image<'static>> {
    let png = unsafe { render_to_png(name, point_size) }?;
    Image::from_bytes(&png).ok()
}

#[allow(deprecated)] // lockFocus / initWithFocusedViewRect — still functional through current SDKs.
unsafe fn render_to_png(name: &str, point_size: f64) -> Option<Vec<u8>> {
    let name_ns = NSString::from_str(name);
    let image: Retained<NSImage> =
        NSImage::imageWithSystemSymbolName_accessibilityDescription(&name_ns, None)?;

    let size = NSSize {
        width: point_size,
        height: point_size,
    };
    image.setSize(size);

    // lockFocus / initWithFocusedViewRect is the simplest off-screen-render
    // route for an NSImage. It's deprecated since macOS 14 but still
    // functional through current SDKs; if Apple ever removes it we'll move
    // to the NSGraphicsContext path.
    image.lockFocus();
    let rect = NSRect {
        origin: NSPoint::ZERO,
        size,
    };
    let rep = NSBitmapImageRep::initWithFocusedViewRect(NSBitmapImageRep::alloc(), rect);
    image.unlockFocus();
    let rep = rep?;

    let props = NSDictionary::new();
    let data = rep.representationUsingType_properties(NSBitmapImageFileType::PNG, &props)?;
    let len = data.length();
    let ptr = data.bytes().as_ptr();
    Some(std::slice::from_raw_parts(ptr, len).to_vec())
}
