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


def test_acp_block_meta_is_preserved_on_attachments() -> None:
    _text, attachments = prompt_parts_from_dict_blocks(
        [
            {
                "type": "image",
                "data": "aW1hZ2U=",
                "mimeType": "image/jpeg",
                "_meta": {
                    "applicationScene": "反光衣/作业服穿戴检测",
                    "objects": [{"label": "no_reflective_vest"}],
                },
            }
        ]
    )

    assert len(attachments) == 1
    assert attachments[0].metadata["applicationScene"] == "反光衣/作业服穿戴检测"
    assert attachments[0].metadata["objects"] == [{"label": "no_reflective_vest"}]
    assert attachments[0].metadata["acp_type"] == "image"


def test_image_block_url_is_treated_as_attachment_path() -> None:
    text, attachments = prompt_parts_from_dict_blocks(
        [
            {"type": "text", "text": "请复判图片"},
            {
                "type": "image",
                "url": "https://example.test/images/review.jpg?token=abc",
                "mimeType": "image/jpeg",
                "_meta": {"reviewSourceId": "source-from-image-block"},
            },
        ]
    )

    assert text == "请复判图片"
    assert len(attachments) == 1
    assert attachments[0].path == "https://example.test/images/review.jpg?token=abc"
    assert attachments[0].mime_type == "image/jpeg"
    assert attachments[0].metadata["reviewSourceId"] == "source-from-image-block"
