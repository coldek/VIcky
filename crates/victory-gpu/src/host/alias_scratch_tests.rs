use super::*;
use crate::object::{S_OK, This, guid, slot};

struct TitleTexture(This);

impl TitleTexture {
    fn new(width: u32, height: u32, format: DXGI_FORMAT, flags: u32) -> Self {
        let desc = [
            3u32,
            0,
            0,
            0,
            width,
            0,
            height,
            1 | 1 << 16,
            format.0 as u32,
            1,
            0,
            0x100,
            flags,
            0,
        ];
        let heap = [1u32, 0, 0, 0, 0];
        let iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut texture = std::ptr::null();
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                heap.as_ptr().cast(),
                0,
                desc.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut texture,
            ),
            S_OK,
        );
        Self(texture)
    }
}

impl Drop for TitleTexture {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

struct Readback {
    buffer: ID3D12Resource,
    pitch: usize,
    rows: u32,
    bytes: usize,
}

impl Readback {
    fn record(host: &Host, list: &HostList, title: &TitleTexture) -> Self {
        let texture = crate::resource::host_resource(title.0).expect("a host texture");
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut total) = (0, 0);
        // SAFETY: a live texture and footprint outputs for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &texture.GetDesc(),
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let buffer = host.read_back_buffer(total).expect("a readback buffer");
        crate::list::ensure_state(list, title.0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        // SAFETY: these copy arguments borrow the live resources. The list
        // tracks the texture's state, and the readback lives through both runs.
        unsafe {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        Self {
            buffer,
            pitch: footprint.Footprint.RowPitch as usize,
            rows,
            bytes: total as usize,
        }
    }

    fn check_rows(&self, pixel: [u8; 4]) {
        let range = D3D12_RANGE {
            Begin: 0,
            End: self.bytes,
        };
        let mut data = std::ptr::null_mut();
        // SAFETY: the finished list copied two RGBA pixels (or one BC1
        // block holding their bytes) into each row of this buffer.
        unsafe {
            self.buffer.Map(0, Some(&range), Some(&mut data)).unwrap();
            for row in 0..self.rows as usize {
                let actual = data
                    .cast::<u8>()
                    .add(row * self.pitch)
                    .cast::<[[u8; 4]; 2]>()
                    .read();
                assert_eq!(actual, [pixel; 2], "row {row}");
            }
            self.buffer.Unmap(0, None);
        }
    }
}

#[test]
fn texture_alias_scratch_reuses_buffers_until_fence_safe_reset() {
    let Some(host) = host() else { return };
    let list = host.new_list().unwrap();
    let rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
    let bc1 = DXGI_FORMAT_BC1_UNORM;
    let mut outputs = Vec::new();
    let incompatible = TitleTexture::new(4, 4, bc1, 0);
    let cases = [
        (2, 2, 2, rgba, [255, 0, 0, 255]),
        (1, 4, 4, bc1, [0, 255, 0, 255]),
        (260, 2, 260, rgba, [0, 0, 255, 255]),
    ];
    for (i, (height, dest_width, dest_height, format, pixel)) in cases.into_iter().enumerate() {
        let title = TitleTexture::new(2, height, rgba, 1);
        let dest = TitleTexture::new(dest_width, dest_height, format, 0);
        let source = crate::resource::host_copy(title.0).expect("a host render target");
        if i == 0 {
            assert!(!crate::texture::alias_copy(
                &list,
                source.clone(),
                incompatible.0
            ));
            assert!(
                list.deferred().alias_scratch.is_none(),
                "different row counts still reject an alias"
            );
        }
        crate::list::ensure_state(&list, title.0, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
        let rtv = host.rtv(Some(&source.texture), None);
        let color = pixel.map(|byte| f32::from(byte) / 255.0);
        // SAFETY: the open list tracks a live render target in that state.
        unsafe { list.list.ClearRenderTargetView(rtv, &color, None) };
        assert!(crate::texture::alias_copy(&list, source, dest.0));
        let scratch = list.deferred().alias_scratch.as_ref().unwrap().0.clone();
        let readback = Readback::record(host, &list, &dest);
        outputs.push((title, dest, readback, pixel, scratch));
    }

    let first = &outputs[0].4;
    let second = &outputs[1].4;
    let grown = &outputs[2].4;
    assert_eq!(
        first.as_raw(),
        second.as_raw(),
        "a second alias with matching capacity reuses the buffer"
    );
    assert_ne!(
        first.as_raw(),
        grown.as_raw(),
        "a larger alias grows the scratch"
    );
    let retained = |buffer: &ID3D12Resource| {
        let identity = buffer.cast::<windows::core::IUnknown>().unwrap();
        list.deferred()
            .kept
            .iter()
            .filter(|object| **object == identity)
            .count()
    };
    assert_eq!(
        retained(first),
        1,
        "the list keeps its replaced buffer once"
    );
    assert_eq!(
        retained(grown),
        0,
        "the active scratch already owns the buffer"
    );

    let other = host.new_list().unwrap();
    let independent = other.alias_scratch(host, 1).unwrap();
    assert_ne!(independent.as_raw(), first.as_raw());
    assert_ne!(independent.as_raw(), grown.as_raw());

    list.close();
    for _ in 0..2 {
        host.execute(&[&list]);
        for (_, _, readback, pixel, _) in &outputs {
            readback.check_rows(*pixel);
        }
    }
    list.reset();
    assert!(list.deferred().alias_scratch.is_none());
    assert_eq!(retained(first), 0, "reset released the replaced buffer");
}
