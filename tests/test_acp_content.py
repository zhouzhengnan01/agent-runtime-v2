from app.protocols.acp.content import prompt_parts_from_dict_blocks


def test_resource_link_url_is_treated_as_attachment_path() -> None:
    text, attachments = prompt_parts_from_dict_blocks(
        [
            {"type": "text", "text": "请复判视频"},
            {
                "type": "resource_link",
                "url": "https://example.test/videos/review.mp4?token=abc",
                "name": "review.mp4",
                "mimeType": "video/mp4",
            },
        ]
    )

    assert text == "请复判视频"
    assert len(attachments) == 1
    assert attachments[0].name == "review.mp4"
    assert attachments[0].path == "https://example.test/videos/review.mp4?token=abc"
    assert attachments[0].mime_type == "video/mp4"
    assert attachments[0].metadata["url"] == "https://example.test/videos/review.mp4?token=abc"
